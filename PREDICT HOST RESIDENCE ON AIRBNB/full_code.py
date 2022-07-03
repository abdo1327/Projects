# import all the dependency needed to run the code from dependency.py
from dependency import * 

#%%
#Empty arrays to store the  accuracy and the  running time for each classifier 
accurcy=[]
runing_time=[] 
#%%  Train and evaluate the RNN classifier 

# for reproducibility  
SEED = 2021
torch.manual_seed(SEED)
#torch.backends.cudnn.deterministic = False
TEXT = data.Field(tokenize='spacy',batch_first=True,include_lengths=True,pad_token='<pad>', unk_token='<unk>')
LABEL = data.LabelField(dtype = torch.float,batch_first=True)
fields = [(None, None), ('text',TEXT),('label', LABEL)]
#Load custom data set
training_data=data.TabularDataset(path = 'myNewData.csv',format = 'csv',fields = fields,skip_header = True)

train_data, valid_data = training_data.split(split_ratio=0.8, random_state = random.seed(SEED))

#initial glove embeddings
TEXT.build_vocab(train_data,min_freq=3,vectors = "glove.6B.100d")
LABEL.build_vocab(train_data)

#check if cuda can be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#set up batch_size
BATCH_SIZE = 64

#load iterator
train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, valid_data),
    batch_size = BATCH_SIZE,
    sort_key = lambda x: len(x.text),
    sort_within_batch=True,
    device = device)

#define hyper parameter
size_of_vocab = len(TEXT.vocab)
embedding_dim = 100
num_hidden_nodes = 32
num_output_nodes = 1
num_layers = 2
bidirection = True
dropout = 0.2

#Instantiate model

model = classifier(size_of_vocab, embedding_dim, num_hidden_nodes,num_output_nodes, num_layers,
                   bidirectional = True, dropout = dropout)

print(model)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

#initialize pre-train embedding
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

print(pretrained_embeddings.shape)

#define optimizer and loss

optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()

#if cuda can be used
model = model.to(device)
criterion = criterion.to(device)

N_EPOCHS = 5
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    #train model
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)

    #evaluate model
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        #torch.save(model.state_dict(), 'saved_weights.pt')
        torch.save(model,'net.pkl')

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

path='net.pkl'
model=torch.load(path)

#inference

nlp = spacy.load('en_core_web_sm')

def predict(model, sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]  #tokenize the sentence
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]          #convert to integer sequence
    length = [len(indexed)]                                    #compute no. of words
    tensor = torch.LongTensor(indexed).to(device)              #convert to tensor
    tensor = tensor.unsqueeze(1).T                             #reshape in form of batch,no. of words
    length_tensor = torch.LongTensor(length)                   #convert to tensor
    prediction = model(tensor, length_tensor)                  #prediction
    return prediction.item()

#%% Predicted the labels using the prediction dataset and RNN classifier
 
predction_set=pd.read_csv ("predict.csv")
t0 = time.time()
for s in list(predction_set['review']):
    predict(model, s)
t1 = time.time()    

#%% Save the  accurcy and the running time for RNN
RNN_runining_time= t1-t0
runing_time.append(RNN_runining_time)
accurcy.append(valid_acc)
#%% Read the dataset again and preprocess it to apply traditional machine learning  
df = pd.read_excel ("labeled dataset.xlsx")
reviews= df[["review","label"]]
X=reviews['review']
Y=reviews["label"]
# preprocess the dataset using preprocess_text function in  preprocessing.py 
vectoriser = TfidfVectorizer(analyzer=preprocess_text)
New_X = vectoriser.fit_transform(X)
# save the TfidfVectorizer as pickle file to ensure having the same 
# dimensionality during the prediction phase
 
pickle.dump(vectoriser.vocabulary_,open("feature.pkl","wb"))
X_train, X_test, y_train, y_test = train_test_split(New_X, Y, test_size=0.2, random_state=123)
#%% Train and evaluate the KNN classifier 
KNN=KNeighborsClassifier()
KNN_parameters = {
    'n_neighbors': (3,5,7,9,11,13,15),
}
gridSearch (KNN, KNN_parameters,X_train,y_train)
best_KNN= KNeighborsClassifier(n_neighbors=15) 
KNN_acc=evaluatetion(best_KNN,X_train, y_train,X_test, y_test)
#%% Predicted the labels using the prediction dataset and KNN classifier

predction_set=pd.read_csv ("predict.csv")
predction_set['review']=predction_set['review'].apply(str)
X2=predction_set['review']
t0_text_processing_time = time.time()
# using the pickle file as the  vocabulary for the new TfidfVectorizer

vectoriser2 = TfidfVectorizer(analyzer=preprocess_text,vocabulary=pickle.load(open("feature.pkl", "rb")))
predctset = vectoriser2.fit_transform(X2)
t1_text_processing_time = time.time() 
text_processing_time= t1_text_processing_time-t0_text_processing_time 
#%% Save the  accurcy and the running time for KNN

Knn_running_time=performance(best_KNN,predctset,text_processing_time)
runing_time.append(Knn_running_time)
accurcy.append(KNN_acc)

#%% Train and evaluate the Random Forest classifier 

RF=RandomForestClassifier(random_state=0)

param_grid = { 
    'n_estimators': [100,200, 500,1000],
    'max_features': ['auto', 'sqrt', 'log2']
}

gridSearch (RF, param_grid,X_train,y_train)

# the best model returned from the gridSearch

best_RF= RandomForestClassifier(random_state=0,max_features='auto',n_estimators=500)
# evaluate the best model

RF_acc=evaluatetion(best_RF,X_train, y_train,X_test, y_test)
   
#%% Predicted the labels by Random Forest classifier and save accurcy and the running time 

RF_running_time=performance(best_KNN,predctset,text_processing_time)
runing_time.append(RF_running_time)
accurcy.append(RF_acc)
#%%  Train and evaluate the SVC classifier 

SVC=SVC(random_state=0)
SVC_parameters = {
    'C': [.1,1,10,100,1000],
    'kernel':('linear', 'rbf','poly')
}

gridSearch (SVC, SVC_parameters,X_train,y_train)
from sklearn.svm import SVC
# the best model returned from the gridSearch

best_SVC= SVC(random_state=0,kernel='linear') 
SVC_acc=evaluatetion(best_SVC,X_train, y_train,X_test, y_test)

#%% Predicted the labels by SVC  classifier and save accurcy and the running time 

SVC_running_time=performance(best_SVC,predctset,text_processing_time)
runing_time.append(SVC_running_time)
accurcy.append(SVC_acc)
#%% Train and evaluate the Decision Tree classifier 

DT=DecisionTreeClassifier(random_state=0)
tree_param = {'criterion':['gini','entropy'],
              'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,100]}

# the best model returned from the gridSearch

gridSearch (DT, tree_param,X_train,y_train)
best_DT=DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=0) 
DT_acc=evaluatetion(best_DT,X_train, y_train,X_test, y_test) 

#%% Predicted the labels by decision Tree classifier and save accurcy and the running time 

DT_running_time=performance(best_DT,predctset,text_processing_time)
runing_time.append(DT_running_time)
accurcy.append(DT_acc)
#%% Train and evaluate the Logistic Regression classifier

LR=LogisticRegression(random_state=0)
LR_param = {'C': [.1,1,10,100,1000],
              'multi_class':('ovr','multinomial')}

# the best model returned from the gridSearch
gridSearch (LR, LR_param,X_train,y_train)
best_LR=LogisticRegression(C=1, multi_class='multinomial', random_state=0)
LR_acc=evaluatetion(best_LR,X_train, y_train,X_test, y_test)  
#%% Predicted the labels by Logistic Regression and save accurcy and the running time 
LR_running_time=performance(best_LR,predctset,text_processing_time)
runing_time.append(LR_running_time)
accurcy.append(LR_acc)
#%% create an data frame
# create an data frame  that has four columns  which are 
# the model name , model's accuracy, running time needed to predict 7.5K row ,
# estimated running time needed to predict   1M row in hours 
columns = ['accurcy']
accurcy_score = accurcy
summary = pd.DataFrame((accurcy_score),columns = columns)
summary['Model'] = ['RNN','KNN','RF','SVM','DT','LR']
runing_time_in_min=[x / 60 for x in runing_time]
summary['Runing time(min) for 7.5K row']=runing_time_in_min
estimated_running_time=[(x*133.333) / 3600 for x in runing_time]
summary['estimated running time for 1M row in hours']=estimated_running_time
summary = summary[['Model', 'accurcy', 'Runing time(min) for 7.5K row', 'estimated running time for 1M row in hours']]
print(summary)

#%% create and plot that shows the running time and the accuracy  for each model 

fig,ax = plt.subplots(figsize=(14,8))
first=ax.plot(summary['Model'], summary['accurcy'], color="#1d3557", marker="o",label='accurcy')
ax.set_xlabel("Model",fontsize=18)
# set y-axis label
ax.set_ylabel("accurcy",fontsize=18)
ax2=ax.twinx()
second=ax2.plot(summary['Model'], summary['Runing time(min) for 7.5K row'],color="#e63946",marker="o",label='Runing time')
ax2.set_ylabel("Runing time(min) for 7.5K row",fontsize=18)
ax.tick_params(axis="x", labelsize=15)
ax.tick_params(axis="y", labelsize=16)
ax2.tick_params(axis="y", labelsize=16)
lns = first+second
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=7,fontsize=16)
plt.show()
# save the plot as a file
fig.savefig('The performance.png',
            format='png',
            dpi=100,
            bbox_inches='tight')
#%%
summary.to_excel('The performance.xlsx')
#%%
plt.figure(2,figsize=(18,8))
X_axis2 = np.arange(len(summary['Model']))
pps=plt.bar(X_axis2 - 0.1, summary['accurcy'], 0.2, label = 'accurcy' ,color='#1d3557', )
plt.bar(X_axis2+ 0.1, summary['Runing time(min) for 7.5K row'], 0.2, label = 'Runing time(min)',color='#e63946')

width=0.2
plt.xticks(X_axis2, summary['Model'])
plt.xlabel("The Model ",fontsize=18,fontweight='bold')
plt.ylabel("The performance",fontsize=18,fontweight='bold')
plt.title(" The Model performance",fontsize=20,fontweight='bold')
plt.xticks(fontsize=16,fontweight='bold',ha="right" )
plt.yticks(fontsize=16 ,fontweight='bold')
plt.legend(loc=0,fontsize=16)

plt.show()