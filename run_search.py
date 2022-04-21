import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from supar import Parser
import string
import random
from nltk.tokenize.treebank import TreebankWordDetokenizer
import numpy as np
import argparse
from nat_inst_gpt3 import *
from sklearn.metrics import balanced_accuracy_score
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from scipy.stats import entropy
import json

parser = argparse.ArgumentParser(description='Take arguments from commandline')
parser.add_argument('--mode', default="Instruction Only", help='Type mode of instructions/prompts')
parser.add_argument('--num-shots', default=2, type=int, help='Type number of examples in the prompt if applicable')
parser.add_argument('--batch-size', default=4, type=int, help='Type in the batch-size')
parser.add_argument('--task-idx', default=1, type=int, help='Type in the index of the task based on the array in the code')
parser.add_argument('--seed', type=int, help='Type in seed that changes sampling of examples')
parser.add_argument('--train-seed', type=int, help='Type in seed that changes the sampling of edit operations (search seed)')
parser.add_argument('--num-compose', default=1, type=int, help='Number of edits composed to get one candidate')
parser.add_argument('--num-train', default=100, type=int, help='Number of examples in score set')
parser.add_argument('--level', default="phrase", help='level at which edit operations occur')
parser.add_argument('--simulated-anneal', action='store_true', default=False, help='runs simulated anneal if candidate scores <= base score')
parser.add_argument('--agnostic', action='store_true', default=False, help='uses template task-agnostic instruction')
parser.add_argument('--print-orig', action='store_true', default=False, help='print original instruction and evaluate its performance')
parser.add_argument('--write-preds', action='store_true', default=False, help='store predictions in a .json file')
parser.add_argument('--meta-dir', default='logs/', help='folder location to store metadata of search')
parser.add_argument('--meta-name', default='search.txt', help='file name to store metadata of search')
parser.add_argument('--patience', default=2, type=int, help='Type in the max patience P (counter)')
parser.add_argument('--num-candidates', default=5, type=int, help='Number of candidates in each iteration (m)')
parser.add_argument('--num-iters', default=10, type=int, help='Max number of search iterations')
parser.add_argument('--key-id', default=0, type=int, help='Use if you have access to multiple Open AI keys')
parser.add_argument('--edits', nargs="+", default=['del', 'swap', 'sub', 'add'], help='space of edit ops to be considered')


args = parser.parse_args()

if args.key_id:
    import nat_inst_gpt3
    nat_inst_gpt3.key = args.key_id

meta_path = os.path.join(args.meta_dir, args.meta_name)
meta_file = open(meta_path, 'w+')
batch_size = args.batch_size
num_shots = args.num_shots
mode = args.mode
seed = args.seed
train_seed = args.train_seed

classification_task_ids = ['019', '021', '022', '050', '069', '137', '139','195']
data_base_path = "data/ExpandedNaturalInstructions/" #location of the Natural Instructions dataset
file_map = {f.split("_")[0]:f for f in os.listdir(data_base_path)}
assert args.task_idx >= 0 and args.task_idx < len(classification_task_ids), "Invalid task index entered."
chosen_task = classification_task_ids[args.task_idx] 
chosen_task_name = file_map['task' + chosen_task]
print("Running Experiment for: ", chosen_task_name)
file_contents = json.load(open("{}/{}".format(data_base_path, chosen_task_name)))
label_list = [file_contents["Instances"][i]["output"][0] for i in range(len(file_contents["Instances"])) ]
num_samples = 100 #default test set of size 100
num_train_samples = args.num_train

np.random.seed(train_seed)
torch.manual_seed(train_seed)
_, task_labels , _ = construct_instruction_prompt(mode='No Instructions', task_name=chosen_task_name, num_shots=num_shots, num_test_instances=num_samples, seed=seed)
task_labels = list(set(task_labels))
task_labels.sort()
print(task_labels)

instruction = file_contents['Definition']
instruction = instruction.replace('\n' + 'Things to avoid: -', '')
instruction = instruction.replace('\n' + 'Emphasis & Caution: -', '')
if args.agnostic:
    instruction = "You will be given a task. Read and understand the task carefully, and appropriately answer '{}' or '{}'.".format(task_labels[0], task_labels[1])
parser = Parser.load('crf-con-en')
num_compose = args.num_compose
num_candidates = args.num_candidates
num_steps = args.num_iter
T_max = 10
edit_operations = args.edits
use_add = 'add' in edit_operations

if 'sub' in edit_operations:
    para_model_name = 'tuner007/pegasus_paraphrase'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    para_tokenizer = PegasusTokenizer.from_pretrained(para_model_name)
    para_model = PegasusForConditionalGeneration.from_pretrained(para_model_name).to(torch_device).eval()


def detokenize(tokens):
    return TreebankWordDetokenizer().detokenize(tokens)


def traverse_tree(parsed_tree):
    phrases = []
    for tree in parsed_tree:
        if tree.label() == '_': continue
        phrases.append(detokenize(tree.leaves()))
        for subtree in tree:
            if type(subtree) == nltk.tree.Tree:
                if subtree.label() == '_': continue
                phrases.append(detokenize(subtree.leaves()))
                phrases.extend(traverse_tree(subtree))
    return phrases

def check_child(tree):
    check = False
    count = 0
    total_count = 0
    for subtree in tree:
        total_count += 1
        if type(subtree) == nltk.tree.Tree:
            if subtree.label() == '_':
                count += 1
    if count >= total_count - count: check = True

    return check

def collect_leaves(parsed_tree):
    leaves = []
    for tree in parsed_tree:
        if type(parsed_tree) != nltk.tree.Tree: continue
        if tree.label() == '_': 
            leaves.append(detokenize(tree.leaves()))
            continue
        if check_child(tree): leaves.append(detokenize(tree.leaves()))
        else:
            leaves.extend(collect_leaves(tree))
    return leaves

def get_phrases(instruction): # one possible way of obtaining disjoint phrases
    phrases = []
    for sentence in sent_tokenize(instruction):
        parsed_tree = parser.predict(word_tokenize(sentence), verbose=False).sentences[0].trees[0]
        leaves = collect_leaves(parsed_tree)
        phrases.extend(leaves)
    phrases = [detokenize(word_tokenize(phrase)) for phrase in phrases if phrase not in string.punctuation or phrase == '']
    return phrases

def get_response(input_text,num_return_sequences,num_beams):
  batch = para_tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = para_model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = para_tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text


def delete_phrase(candidate, phrase):
    if candidate.find(' ' + phrase) > 0:
        answer = candidate.replace(' ' + phrase, ' ')
    elif candidate.find(phrase + ' ') > 0:
        answer = candidate.replace(phrase + ' ', ' ')
    else: 
        answer = candidate.replace(phrase, '')
    return answer

def add_phrase(candidate, phrase, after):
    if after == '': answer = phrase + ' ' + candidate
    else: 
        if candidate.find(' ' + after) > 0:
            answer = candidate.replace(' ' + after, ' ' + after + ' ' + phrase)
        elif candidate.find(after + ' ') > 0:
            answer = candidate.replace(after + ' ', after + ' ' + phrase + ' ')
        else: 
            answer = candidate.replace(after, after + phrase )
    return answer

def swap_phrases(candidate, phrase_1, phrase_2):
    if candidate.find(' ' + phrase_1 + ' ') >= 0 : 
        answer = candidate.replace(' ' + phrase_1 + ' ', ' <1> ')
    else: answer = candidate.replace(phrase_1, '<1>')
    if candidate.find(' ' + phrase_2 + ' ') >= 0 : 
        answer = candidate.replace(' ' + phrase_2 + ' ', ' <2> ')
    else: answer = candidate.replace(phrase_2, '<2>')
    answer = answer.replace('<1>', phrase_2)
    answer = answer.replace('<2>', phrase_1)
    return answer

def substitute_phrase(candidate, phrase):
    num_beams = 10
    num_return_sequences = 10
    paraphrases = get_response(phrase, num_return_sequences, num_beams)
    paraphrase = np.random.choice(paraphrases, 1)[0] 
    paraphrase = paraphrase.strip('.')
    if candidate.find(' ' + phrase) > 0:
        answer = candidate.replace(' ' + phrase, ' ' + paraphrase)
    elif candidate.find(phrase + ' ') > 0:
        answer = candidate.replace(phrase + ' ', paraphrase + ' ')
    else: 
        answer = candidate.replace(phrase, paraphrase)
    return answer

def perform_edit(edit, base, phrase_lookup, delete_tracker):
    if edit == 'del':
        [i] = np.random.choice(list(phrase_lookup.keys()), 1) 
        return delete_phrase(base, phrase_lookup[i]), [i]
    elif edit == 'swap':
        try: [i, j] = np.random.choice(list(phrase_lookup.keys()), 2, replace=False) 
        except: [i, j] = np.random.choice(list(phrase_lookup.keys()), 2, replace=True) 
        return swap_phrases(base, phrase_lookup[i], phrase_lookup[j]), [i, j]
    elif edit == 'sub':
        [i] = np.random.choice(list(phrase_lookup.keys()), 1) 
        return substitute_phrase(base, phrase_lookup[i]), [i]
    elif edit == 'add':
        keys = list(phrase_lookup.keys())
        keys.append(-1)
        [i] = np.random.choice(keys, 1) 
        if i >= 0: after = phrase_lookup[i]
        else: after = ''
        if len(delete_tracker) == 0: return base, []
        phrase = np.random.choice(delete_tracker, 1)[0]
        return add_phrase(base, phrase, after), [phrase]

def custom_instruction_prompt(mode=mode, task_name=chosen_task_name, num_shots=num_shots, num_test_instances=num_samples, seed=seed, null_word=None, split='train', modified={}):
    if mode=="Instruction Only":
        prompt_list, answer_list, index_list, train_prompt_list, train_answer_list, train_index_list, dev_prompt_list, dev_answer_list, dev_index_list = training_encodeinstruction(task_name, instruction_structure = ["Definition"], number_of_examples = num_shots, number_of_instances = num_test_instances, seed=seed, null_word=null_word, modified=modified)
    elif mode=="Instruction + Positive Examples":
        prompt_list, answer_list, index_list, train_prompt_list, train_answer_list, train_index_list, dev_prompt_list, dev_answer_list, dev_index_list = training_encodeinstruction(task_name, instruction_structure = ["Definition", "Positive Examples Full Only"], number_of_examples = num_shots, number_of_instances = num_test_instances, seed=seed, null_word=null_word, modified=modified)
    else: raise ValueError()
    if split == 'test': return prompt_list, answer_list, index_list
    elif split == 'train': 
        train_prompt_list.extend(dev_prompt_list)
        train_answer_list.extend(dev_answer_list)
        train_index_list.extend(dev_index_list)
        try:
            random.seed(seed)
            indices = random.sample(range(len(train_index_list)), num_train_samples) 
            train_prompt_list = [train_prompt_list[i] for i in indices]
            train_answer_list = [train_answer_list[i] for i in indices]
            train_index_list = [train_index_list[i] for i in indices]
        except: pass
        
        return train_prompt_list, train_answer_list, train_index_list

    else: raise ValueError()

def score(candidate, split='train', write=False):
    
    label_probs, calibrated_label_probs , raw_acc_count , raw_cal_acc_count , answer_list, index_list, _ = run(mode=mode, batch_size=batch_size, num_shots=num_shots, chosen_task_name=chosen_task_name, num_samples=num_samples, seed=seed, override_prompts=True, function = custom_instruction_prompt, split=split, modified={'Definition': candidate}, task_labels=task_labels, if_calibrate = False)
    preds = get_prediction(label_probs, task_labels)
    raw_acc = balanced_accuracy_score(answer_list, preds)
    label_frequencies = [preds.count(l)/len(preds) for l in task_labels]
    if split == 'train': return np.round(100*raw_acc, 2) + 10*entropy(label_frequencies)
    elif split== 'test': 
        if write:
            pname = args.meta_name
            pname = pname.split('.')[0] + "_predictions.json"
            pred_dump = {'predictions': preds, 'answers': answer_list, 'ids':index_list}
            ppath = os.path.join(args.meta_dir, pname)
            pfile = open(ppath, 'w+')
            json.dump(pred_dump, pfile)
        return np.round(100*raw_acc_count/len(answer_list), 2)
    else: return

def get_phrase_lookup(base_candidate):
    if args.level == 'phrase': phrase_lookup = {p:phrase for p, phrase in enumerate(get_phrases(base_candidate))}
    elif args.level == 'word': 
        words = word_tokenize(base_candidate)
        words = [w for w in words if w not in string.punctuation or w != '']
        phrase_lookup = {p:phrase for p, phrase in enumerate(words)}
    elif args.level == 'sentence':
        sentences = sent_tokenize(base_candidate)
        phrase_lookup = {p:phrase for p, phrase in enumerate(sentences)}
    elif args.level == 'span':
        phrases = []
        for sentence in sent_tokenize(base_candidate):
            spans_per_sentence = np.random.choice(range(2,5)) # split sentence into 2, 3, 4, 5 chunks
            spans = np.array_split(word_tokenize(sentence), spans_per_sentence)
            spans = [detokenize(s) for s in spans]
            phrases.extend(spans)
        phrase_lookup = {p:phrase for p, phrase in enumerate(phrases)}
    else: raise ValueError()
    return phrase_lookup
               

operations_tracker = []
base_candidate = detokenize(word_tokenize(instruction))
assert word_tokenize(base_candidate) == word_tokenize(instruction)
original_candidate = base_candidate
meta_file.write("Base Candidate:\t "+ original_candidate + '\n')
base_score = score(base_candidate)
meta_file.write("Base Score:\t "+ str(base_score) + '\n')
meta_file.write("\n")
delete_tracker = []
patience_counter = 1
for i in range(num_steps):
    meta_file.write("Running step:\t " + str(i) + '\n')
    deleted = {}
    added = {}
    phrase_lookup = get_phrase_lookup(base_candidate)
    if base_candidate == original_candidate:
        for p in phrase_lookup.values(): print(p)
    if use_add: 
        if len(delete_tracker): 
            if 'add' not in edit_operations: edit_operations.append('add')
        else: 
            if 'add' in edit_operations: edit_operations.remove('add')
    if num_compose == 1:
        edits = np.random.choice(edit_operations, num_candidates)
    else: 
        edits = []
        for n in range(num_candidates):
            edits.append(np.random.choice(edit_operations, num_compose))
    print(edits)


    # generate candidates
    candidates = []
    for edit in edits:
        if isinstance(edit, str): 
            meta_file.write("Performing edit:\t "+ edit + '\n')
            candidate, indices = perform_edit(edit, base_candidate, phrase_lookup, delete_tracker)
            meta_file.write("Generated candidate:\t "+ candidate + '\n')
            candidates.append(candidate)
            if edit  == 'del': deleted[candidate] = [phrase_lookup[indices[0]]]
            if edit == 'add': 
                if len(indices): added[candidate] = indices
        else:
            meta_file.write(("Performing edit:\t "+ ' '.join(edit))+ '\n')
            old_candidate = base_candidate
            composed_deletes = []
            composed_adds = []
            for op in edit:
                phrase_lookup = get_phrase_lookup(old_candidate)
                new_candidate, indices = perform_edit(op, old_candidate, phrase_lookup, delete_tracker)
                if op  == 'del':  composed_deletes.append(phrase_lookup[indices[0]])
                if op == 'add': 
                    if len(indices): composed_adds.append(indices[0])
                old_candidate = new_candidate
            meta_file.write("Generated candidate:\t "+ new_candidate+ '\n')
            candidates.append(new_candidate)
            if 'del' in edit: deleted[new_candidate] = composed_deletes
            if 'add' in edit and len(composed_adds) > 0: added[new_candidate] = composed_adds

    
    print(base_score)
    scores = []
    for c, candidate in enumerate(candidates):
        scores.append(score(candidate))
        print(scores[-1])
        meta_file.write("Score for Candidate "+ str(c)+ ":\t "+ str(scores[-1])+ '\n')
    
    meta_file.write("\n")
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    if best_score > base_score: 
        patience_counter = 1
        base_candidate = candidates[best_idx]
        base_score = best_score
        operations_tracker.append(edits[best_idx])
        meta_file.write("New Candidate Found"+ '\n')
        meta_file.write("New Candidate Index:\t "+ str(best_idx)+ '\n')
        meta_file.write("New Candidate:\t "+ base_candidate+ '\n')
        meta_file.write("New Candidate Score:\t "+ str(base_score)+ '\n')
        try: meta_file.write("New Candidate Edit:\t "+ edits[best_idx]+ '\n')
        except: meta_file.write("New Candidate Edit:\t "+ ' '.join(edits[best_idx])+ '\n')
        meta_file.write("\n")
        print('New Base Candidate: ', base_candidate)
        if base_candidate in added.keys():
            print('Notice! Prev tracker: ', delete_tracker)
            for chunk in added[base_candidate]: 
                try: delete_tracker.remove(chunk)
                except: pass
            print('Notice! New tracker: ', delete_tracker)
        if base_candidate in deleted.keys():
            delete_tracker.extend(deleted[base_candidate])
        base_candidate = detokenize(word_tokenize(base_candidate))
        
    else: 
        patience_counter += 1
        
        if args.simulated_anneal:
            K = 5
            T = T_max * np.exp(-i/K)
            idx = np.argmax(scores)
            chosen_score = scores[idx]
            prob = np.exp((chosen_score - base_score)/ T)
            if np.random.binomial(1, prob): 
                print('\n')
                print('Update from simulated anneal')
                meta_file.write('Update from simulated anneal \n')
                base_candidate = candidates[idx]
                base_score = chosen_score
                print('New Base Candidate: '+ base_candidate)
                if base_candidate in added.keys():
                    print('Notice! Prev tracker: ', delete_tracker)
                    for chunk in added[base_candidate]: 
                        try: delete_tracker.remove(chunk)
                        except: pass
                    print('Notice! New tracker: ', delete_tracker)
                if base_candidate in deleted.keys():
                    delete_tracker.extend(deleted[base_candidate])
                base_candidate = detokenize(word_tokenize(base_candidate))
            else:
                if patience_counter > args.patience:
                    print('Ran out of patience')
                    meta_file.write('Ran out of patience \n')
                    break
                else: continue        
            

        else:
            if patience_counter > args.patience:
                print('Ran out of patience')
                meta_file.write('Ran out of patience \n')
                break
            else: continue      
            
meta_file.write('\n')
print('\nTesting .... ')
meta_file.write('Testing .... \n')
if args.print_orig:
    print('Task:\t', chosen_task_name)
    print('Original Instruction:\t', original_candidate)
    orig_score = score(original_candidate, 'test')
    print('Original Accuracy:\t', str(orig_score))

if base_candidate == original_candidate: 
    print('No viable candidate found!')
    meta_file.write('No viable candidate found!\n')
    exit()
searched_score = score(base_candidate, 'test', write=args.write_preds)
print('Accuracy after search:\t', str(searched_score))
print('Instruction after search:\t', base_candidate)
print('Edit Operations:\t', operations_tracker)
meta_file.write('Instruction after search:\t'+ base_candidate+ '\n')
meta_file.write('Accuracy after search:\t'+ str(searched_score)+ '\n')
meta_file.write('Edit Operations:\t'+ ' '.join([str(o) for o in operations_tracker]) + '\n')


