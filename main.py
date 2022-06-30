from __future__ import print_function
import os
import subprocess
import re
import SPARQLWrapper
import json
import parmap
import numpy as np
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import threading
import time
from multiprocessing import Pool
import logging

##################### Just defining some variables and defaults no need to edit anything here please go straight to the main function at the end ########################################
def setParam(P_threadcount = 24, P_split = 1000, P_sparql_endpoints = 4,  P_prefix = 'http://dbpedia.org/ontology/' ,  P_relation = 'owl:disjointWith', P_path = '',
             P_corese_path = os.path.normpath(r""), 
             P_rdfminer_path = os.path.normpath(r""),
             P_command_line = 'start /w cmd /k java -jar -Dfile.encoding=UTF8 -Xmx20G corese-server-4.3.0.jar -e -lp -debug -pp profile.ttl', P_dataset = 'dbpedia_updated_disjoint.owl',
             P_wds_Corese = 'http://localhost:8080/sparql', P_label_type = 'c', P_list_of_axioms = None, P_score = None,  P_dont_score = True, P_set_axiom_number = 0):
    global threadcount    #number of process for multiprocessing avoid using logical cores
    global split          # divide the table you are working on into tasks, the more processors the more you can divide
    global prefix         #use this to reduce the search time and make thigs more readable
    global relation       #the axiom/relation we are extracting
    global path           #the path of kernel builder should be edited in the .py file itself
    global corese_path    #parameters to launch corese server
    global command_line
    global wds_Corese
    global allrelations  #the whole axiom dataset we are working with
    global label_type    #either classification of regression so either a score or a binary label
    global axiom_type    # disjoint, subclass, equivilent or same as 
    global list_of_axioms
    global score
    global list_df
    global rdfminer_path
    global dont_score
    global set_axiom_number
    global sparql_endpoints
    global dataset
    list_df = []
    dataset = P_dataset
    sparql_endpoints = P_sparql_endpoints
    dont_score = P_dont_score
    threadcount = P_threadcount
    split = P_split
    prefix = P_prefix
    relation = P_relation
    path = P_path
    corese_path = P_corese_path
    command_line = P_command_line
    wds_Corese = P_wds_Corese
    label_type = P_label_type
    rdfminer_path = P_rdfminer_path
    if P_relation == 'owl:disjointWith':
        axiom_type = 'DisjointClasses'
    elif P_relation == 'rdfs:subClassOf':
        axiom_type = 'SubClassOf'
    set_axiom_number = P_set_axiom_number
    list_of_axioms = P_list_of_axioms
    score = P_score
    

# Read a list of axioms, extract unique concepts to use in creating a precise concept similarity matrix (finish axiom types)
def clean_scored_atomic_axioms_simple(labeltype = 'c', axiomtype = "SubClassOf", score_ = None,sample = True):
    
    valid = {'c', 'r'}
    if labeltype not in valid:
        raise ValueError("labeltype must be one of %r." % valid)
           
    scored_axiom_list = pd.read_csv(score_, header = 0)
    scored_axiom_list['left'], scored_axiom_list['right'] = zip(*(s.split(" ") for s in scored_axiom_list.iloc[:, 0].apply(lambda x: x.replace('DisjointClasses','')).apply(lambda x: re.sub( '[()]','',x)).apply(lambda x: x.replace('<','"')).apply(lambda x: x.replace('>','"')).apply(lambda x: x.replace('SubClassOf','')) ))
    scored_axiom_list = scored_axiom_list[scored_axiom_list.left != scored_axiom_list.right].reset_index(drop = True)

    if axiomtype == "DisjointClasses":
        a, b = zip(*(s.split(" ") for s in scored_axiom_list.iloc[:, 0].apply(lambda x: x.replace('DisjointClasses','')).apply(lambda x: re.sub( '[()]','',x)).apply(lambda x: x.replace('<','"')).apply(lambda x: x.replace('>','"')) ))
    else:
        a, b = zip(*(s.split(" ") for s in scored_axiom_list.iloc[:, 0].apply(lambda x: x.replace('SubClassOf','')).apply(lambda x: re.sub( '[()]','',x)).apply(lambda x: x.replace('<','"')).apply(lambda x: x.replace('>','"')) ))
    
    #list of all unique concepts in our axiom set
    concepts =  pd.Series(pd.Series(np.hstack([a,b])).drop_duplicates().values).sort_values()
    
    a= pd.Series(a).apply(lambda x: x.replace('"',''))
    b = pd.Series(b).apply(lambda x: x.replace('"',''))
    
    # extract the score for regression and a label for classification based on the type of axiom
    labeled_axioms =  pd.concat([a,b,scored_axiom_list['label']],axis = 1, keys = ["left","right","label"])
    #create the list of axioms to be sent in the query
    concept_string = ",".join(concepts) 
    return concepts,concept_string, labeled_axioms


#corese_server = subprocess.Popen(command_line, shell=True, cwd=corese_path)
def sparql_service_to_dataframe(service, query):
    """
    Helper function to convert SPARQL results into a Pandas DataFrame.

    Credit to Ted Lawless https://lawlesst.github.io/notebook/sparql-dataframe.html
    """
    sparql = SPARQLWrapper(service)
    sparql.setMethod('POST')
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    result = sparql.query()

    processed_results = json.load(result.response)
    cols = processed_results['head']['vars']

    out = []
    for row in processed_results['results']['bindings']:
        item = []
        for c in cols:
            item.append(row.get(c, {}).get('value'))
        out.append(item)

    return pd.DataFrame(out, columns=cols)


#build the classes similarity table
def buildClassSim(listofconcepts = None):
    print('Creating concept similarity matrix')
    if listofconcepts == None:
        query = '''
    
        select * (kg:similarity(?class1, ?class2) as ?similarity)  where {
        ?class1 a owl:Class
        ?class2 a owl:Class
        filter (!isBlank(?class1)  && !isBlank(?class2) && (?class1 <= ?class2))
        }
    
        '''
    else:
        query = '''
    
        select * (kg:similarity(?class1, ?class2) as ?similarity)  where {
        ?class1 a owl:Class
        ?class2 a owl:Class
        filter (!isBlank(?class1)  && !isBlank(?class2) && (?class1 <= ?class2) && str(?class1) IN ('''+ listofconcepts + ''') && str(?class2) IN ('''+ listofconcepts + '''))
        
        }
    
        '''
    tic = time.perf_counter()
    df = sparql_service_to_dataframe(wds_Corese, query)
    df = df.astype({'similarity': 'float'})
    df1 = df[["class2","class1","similarity"]]
    df1 = df1[df1['class1'] != df1['class2']]
    df1.rename(columns={'class2': 'class1', 'class1': 'class2'}, inplace=True)
    df = pd.concat([df, df1], axis=0).reset_index(drop=True)
    
    dfdic = dict(zip(df.class1 + "," +df.class2, df.similarity))
    print(df.shape)
    # create the table of similarity between all the classes
    toc = time.perf_counter()
    print(f"it took {toc - tic:0.4f} seconds to creat concept similarity matrix")
    df.to_csv( path + 'classessim.csv', index=False)
    print('file classsim created and saved')
    print(df.shape)
    
    return df, dfdic


    
#function used to calculate the kernel matrix can be used with fractions of the relationship table


def matrixfractionAverageSimdisdic(start, end, size, df, allrelations):#the column name for the first column is class2
    rowlist = []
    for i in range(start, end):
        axiom1 = allrelations.iloc[i]
        a1_l =  axiom1['left']
        a1_r = axiom1['right']
        for j in range(i, size):
            axiom2 = allrelations.iloc[j]
            #because in disjointness left and right dont make a difference, we compare as dis(A B) dis(B A) and dis(B A) dis(B A)
            sim1 = df[a1_l+","+axiom2['left']]
            sim2 = df[a1_r+","+axiom2['right']]
            
            sim3 = df[a1_l+","+axiom2['right']]
            sim4 = df[a1_r+","+axiom2['left']]
            if (sim1+sim2)/2 > (sim3+sim4)/2:
                sim = (sim1+sim2)/2
            else:
                sim = (sim3+sim4)/2
            rowlist.append([i, j, sim])
    axiomsimilaritymatrix = pd.DataFrame(rowlist, columns=["axiom1", "axiom2", "overallsim"])
    return axiomsimilaritymatrix


def matrixfractionAverageSimdic(start, end, size, df, allrelations):#the column name for the first column is class2
    rowlist = []
    for i in range(start, end):
        axiom1 = allrelations.iloc[i]
        a1_c1 =  axiom1['left']
        a1_c2 = axiom1['right']
        for j in range(i, size):
            axiom2 = allrelations.iloc[j]
            sim1 = df[a1_c1+","+axiom2['left']]
            sim2 = df[a1_c2+","+axiom2['right']]
            rowlist.append([i, j, (sim1+sim2)/2])
    axiomsimilaritymatrix = pd.DataFrame(rowlist, columns=["axiom1", "axiom2", "overallsim"])
    return axiomsimilaritymatrix

# preparing to split work load to multiple threads
def splitload(split, relations):
    size = len(relations)
    portion = size//split
    startend = []
    l = 0
    for x in range(1,split):
        startend.append((l,l+portion))
        l += portion
    startend.append((startend[len(startend)-1][1], size))
    print("split completed")
    return(startend)


def pivotIntofinalmatrix(kernelmatrix):
    tic = time.perf_counter()

    kernelmatrix = kernelmatrix.pivot_table(columns='axiom1', index='axiom2', values='overallsim',  fill_value=0).reset_index()
    kernelmatrix.drop(columns = ["axiom2"], inplace = True)
    rawmatrix = kernelmatrix.to_numpy()
    rawmatrix = rawmatrix + rawmatrix.T - np.diag(np.diag(rawmatrix))
    #added <> and axiom type to axioms to comply with rdf miner fromat
    colnames = axiom_type+"(<"+ allrelations["left"] + "> <" + allrelations["right"] + ">)"
    kernelmatrix = pd.DataFrame(data=rawmatrix, columns = colnames)
    kernelmatrix.insert(0,'possibility',allrelations['label'])
    print("kernelmatrix shape is :") 
    print(kernelmatrix.shape)
    kernelmatrix.to_csv( path + 'kernelmatrix.csv', sep=',', index=False)
    print('file kernelmatrix built and saved')
    toc = time.perf_counter()
    print(f"it took {toc - tic:0.4f} seconds")
    return kernelmatrix

def BuildProfile(P_sparql_endpoints,P_dataset):
    profile = '''
        st:user a st:Server;
                st:content st:load.
    

    '''
    T_dataset = '''
        st:load a st:Workflow;
            sw:body (
        [a sw:Load; sw:path <'''+ P_dataset +'''> ]
        ).
    '''
    profile = profile + T_dataset
    profile_file = open(corese_path + "/profile.ttl", "w")
    n = profile_file.write(profile)
    profile_file.close()
    
def QueryBuilder(P_concept_string, P_concepts):
    queries = []
    chunked_list = np.array_split(P_concepts,sparql_endpoints)
    for i in range(sparql_endpoints):
        query = '''
    
        select * (kg:similarity(?class1, ?class2) as ?similarity) where {
        ?class1 a owl:Class
        ?class2 a owl:Class
        filter (!isBlank(?class1)  && !isBlank(?class2) && (?class1 <= ?class2) && str(?class1) IN ('''+  ",".join(chunked_list[i]) + ''') && str(?class2) IN ('''+ P_concept_string + '''))
        
        }
    
        '''
 
        queries.append(query)
    return queries



def ExecQuery(P_endpoint, P_query, list_df):

    logging.info("Thread %s: starting", P_endpoint)
    df = sparql_service_to_dataframe('http://localhost:8080/sparql', P_query)
    df = df.astype({'similarity': 'float'})
    df1 = df[["class2","class1","similarity"]]
    df1 = df1[df1['class1'] != df1['class2']]
    df1.rename(columns={'class2': 'class1', 'class1': 'class2'}, inplace=True)
    df = pd.concat([df, df1], axis=0).reset_index(drop=True)
    dfdic = dict(zip(df.class1 + "," +df.class2, df.similarity))
    list_df.append(dfdic)
    logging.info("Thread %s: finishing %s", P_endpoint, time.time() - start)


def buildConceptsimthreaded(P_queries):
    ticfirst = time.perf_counter()
    threads = []
    for i in range(len(P_queries)):#changed from sparql endpoints to length of queries
        threads.append(threading.Thread(target=ExecQuery, args=(i, P_queries[i],list_df)))
    for x in threads:
        x.start()
    for x in threads:
        x.join()
    dfdic = {k:v for x in list_df for k,v in x.items()}
    print(str(len(dfdic))+"this is the length of df dic")
    tocfirst = time.perf_counter()
    print(f"it took {tocfirst - ticfirst:0.4f} seconds for threaded concept similarity")
    return dfdic




def get_concept_relation_list(selected_feature):
    if axiom_type == "DisjointClasses":
        a, b = zip(*(s.split(" ") for s in pd.Series(selected_feature).apply(lambda x: x.replace('DisjointClasses','')).apply(lambda x: re.sub( '[()]','',x)).apply(lambda x: x.replace('<','"')).apply(lambda x: x.replace('>','"')) ))
    else:
        a, b = zip(*(s.split(" ") for s in pd.Series(selected_feature).apply(lambda x: x.replace('SubClassOf','')).apply(lambda x: re.sub( '[()]','',x)).apply(lambda x: x.replace('<','"')).apply(lambda x: x.replace('>','"')) ))
    
    #list of all unique concepts in our axiom set
    concepts =  pd.Series(pd.Series(np.hstack([a,b])).drop_duplicates().values)
    a= pd.Series(a).apply(lambda x: x.replace('"',''))
    b = pd.Series(b).apply(lambda x: x.replace('"',''))

    relations =  pd.concat([a,b],axis = 1, keys = ["left","right"])
    return(concepts,relations)
     


# WARNING CHANGE THE SLEEP TIMER AFTER LAUNCHING CORESE IF YOU ARE HAVING AN ERROR WHEN THE OWL FILE IS LARGE, 50 SECONDS IS GOOD FOR 200 MB FILES, 10 IS GOOD FOR 8 MB
#calls for multiprocessing to build the table of axiom similarity
if __name__ == '__main__':
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
    ticstart = time.perf_counter()
    ticfirst = time.perf_counter()
    pd.set_option('display.max_colwidth', None) # if your Pandas version is < 1.0 then use -1 as second parameter, None otherwise
    pd.set_option('display.precision', 5)
    pd.set_option('display.max_rows', 99999999999)
    #end version, every parameter that can be changed should be here
    setParam(P_threadcount = 6, P_split =16,  P_prefix = '' , P_sparql_endpoints =16, P_dataset = 'dbpedia_updated_disjoint_addedconcepts.owl',##### This file is provided and should be placed in the Corese directory
             P_path = '',
             P_corese_path = os.path.normpath(""),########### important change this path accordingly to where you corese server is installed, keep it empty if corese is in the same directory
             P_command_line = 'start /w cmd /k java -jar -Dfile.encoding=UTF8 -Xmx24G corese-server-4.3.0.jar -e -lp -pp profile.ttl', 
             P_wds_Corese = 'http://localhost:8080/sparql', 
             P_relation = 'owl:disjointWith',########################## type of axiom, please write either owl:disjointWith or rdfs:subClassOf
             P_label_type='r', 
             P_list_of_axioms= None, 
             P_score = "scored reg.txt", ####################### path to the file containing scored set of axioms, can be scored by any scorer
             P_dont_score = True,
             P_set_axiom_number =4000)

    #prepare to launch corese server
    BuildProfile(sparql_endpoints,dataset)
    corese_server = subprocess.Popen(command_line, shell=True, cwd=corese_path)
    
    
    #####################################CHANGE THIS TIMER IN CASE OF ERRORS
    time.sleep(6)

    
   
    print('using a scored list of axioms')
    concepts, concept_string, allrelations = clean_scored_atomic_axioms_simple(label_type, axiom_type, score)
    concepts = concepts.sample(frac = 1)
    queries = QueryBuilder(concept_string,concepts)#split concepts into multiple queries to make the process threaded
    time.sleep(1)
    start = time.time()
    dfdicbase = buildConceptsimthreaded(queries)#start the threaded query process



    #split load into multiple processes and list of axioms into chunks
    startend = splitload(split, allrelations)
    size = len(allrelations)
    p = Pool(threadcount)
    tocfirst = time.perf_counter()
    print(f"it took {tocfirst - ticfirst:0.4f} seconds")
    print()
    
    
    tic = time.perf_counter()

    if axiom_type == 'DisjointClasses':
        print('mirror compare')
        kernelmatrix = pd.concat(parmap.starmap(matrixfractionAverageSimdisdic,startend,size, dfdicbase, allrelations, pm_pool=p, pm_pbar=True),ignore_index = True)
    #similarity is averag
    else:   
        kernelmatrix = pd.concat(parmap.starmap(matrixfractionAverageSimdic,startend,size, dfdicbase, allrelations, pm_pool=p, pm_pbar=True),ignore_index = True)
    
    tocc = time.perf_counter()
    print(f"axiom sim took {tocc - tic:0.4f} seconds")
    p.close()
    p.terminate()
    p.join()
    
    #turn the list into a matrix
    kernelmatrix = pivotIntofinalmatrix(kernelmatrix)
    #finished creating data set
    ################################################################################################################################
    