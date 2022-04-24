
import wikipedia as w

w.set_lang('en')

import re

import nltk

import string

import random

from pathos.threading import ThreadPool as Pool

def clean_punctuation_from_tokenized(word_list):
    return [w for w in word_list if w and not w in string.punctuation]

def clean_and_tokenize_text(text):
    t1 = re.sub("\[.*\]", '', text) # [data in square brackets]
    t2 = re.sub('\s+',' ', t1) # filter text from whitespaces
    
    # specific cleaning
    t2 = t2.replace("``","")
    t2 = t2.replace("''","")
    
    
    #t2 = re.sub("=.*=", '', t2) # specific for wikipedia, thins like "=== Indices ==="
    
    sentences = nltk.sent_tokenize(t2)
    # filter punctuation. The problem are sentences where there is no space after comma, such thins will be left as 'words':
    # 'I like jam.I eat it a lot --> ['I', 'like', 'jam.I', 'eat', 'it', 'a', 'lot']
    fully_tokenized = [clean_punctuation_from_tokenized(nltk.word_tokenize(sentence)) for sentence in sentences]
    return fully_tokenized
    
    
    
    

# TODO: search for "Company..." in w.categories, sometimes forst result is not company. Plus, add "company" to searches

# get first page from wikipedia search results which in theory should correspond to entity
def get_wiki_page(entity, ent_type):
    #Grab  the list from wikipedia.
    # we iterate three top results from search to find company
    found = False
    for i in range(3):
        try:
            s = w.page(w.search(entity + ' ' + ent_type)[i])
        except:
            return None, None        
        for c in s.categories:
            if c.lower().startswith("companies"):
                found = True
                break
        if found:
            break
    if found:
        content = s.content # Content of page.
        title = s.title
        return content, title
    return None, None
# example:
#d, t = get_wiki_page('BP','corporation')



# get data for corporations list. Save all the data in the train file.
def generate_traning_data_for_entity(companies, ent_name='corporation'):
    
            
    def process_company(company):   
        res = []
        counterexample_sentences = []
        
        company = company.strip()
        # get wiki page for company
        d, name = get_wiki_page(company, ent_name)
        if not d:
            return None
        tokenized = clean_and_tokenize_text(d)
        # choose only the sentences which contain company name:
        
        lcomp = company.lower()
        lname = name.lower()
        for sentence in tokenized:
            sent = [s.lower() for s in sentence]
            snt_joined = ' '.join(sent)
            # important to remove here, otherwise a lot of data will be lost
            snt_joined = re.sub("=.*=", '', snt_joined)
            sent = [w for w in snt_joined.split() if w]
            if sent:
                if (lname in sent):
                    # split sent by this element
                    pos = sent.index(lname)
                    sent1 = ' '.join(sent[:pos])
                    sent2 = ' '.join(sent[pos+1:])
                    
                    res.append([lname, sent1, sent2])
                elif (lcomp in sent):
                    pos = sent.index(lcomp)
                    sent1 = ' '.join(sent[:pos])
                    sent2 = ' '.join(sent[pos+1:])                
                    res.append([lcomp, sent1, sent2])
                else:
                    # pick random word which is not company name
                    if len(sent)>2:
                        cword = random.choice(sent[1:-1])
                    else:
                        cword = random.choice(sent)
                    pos = sent.index(cword)
                    sent1 = ' '.join(sent[:pos])
                    sent2 = ' '.join(sent[pos+1:])
                    
                    counterexample_sentences.append([cword, sent1, sent2])
                
        print(company)  
        
        result = [res,counterexample_sentences]
        return result
    
    pool = Pool()
    res = list(pool.imap(process_company, companies))
    pool.close()
    pool.join()
        
    pool.terminate() # needed for pathos https://stackoverflow.com/questions/49888485/
    pool.restart()
    
    # create tab-separated data: <name1> \t <name2> \t <sentence>
    result = []
    counter_result = []
    
    list_of_lists = [r for r in res if r]
    for item in list_of_lists:
        
        for positive in item[0]:
            r = positive[0] + '\t' + positive[1] + '\t' + positive[2]
            result.append(r)
        for negative in item[1]:
            r = negative[0] + '\t' + negative[1] + '\t' + negative[2]
            counter_result.append(r)
    
    return result, counter_result 
    
# read company files
companies = []
with open('corporations.txt', 'r') as f:
    for company in f:
        company = company.strip()
        companies.append(company.strip(string.punctuation))
        
with open('constituents.csv', 'r') as f:
    for line in f:
        field2 = line.split(",")[1:-1]
        field2 = ",".join(field2)
        field2 = field2.replace("\"","").replace("\'","")
        field2 = field2.strip()
        companies.append(field2.strip(string.punctuation))
        
companies = sorted(list(set(companies)))

print("%d unique companies" % len(companies))
        
    
examples_positive, examples_negative = generate_traning_data_for_entity(companies, ent_name='corporation')

with open("positive.txt", "w") as f:
    f.writelines( "%s\n" % item for item in examples_positive )

with open("negative.txt", "w") as f:
    f.writelines( "%s\n" % item for item in examples_negative )
        
        



