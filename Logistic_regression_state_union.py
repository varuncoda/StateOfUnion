from nrclex import NRCLex
import nltk
nltk.download('punkt')
nltk.download('stopwords')
#nltk.download('omw-1.4')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
#import re   


def sentiment(file,word):
    if word=='n':

        x=open(file, encoding="utf8")
        text=x.read()
        sentence=nltk.sent_tokenize(text)

        word=nltk.word_tokenize(text)
        from nltk.corpus import stopwords
        stop_words=set(stopwords.words("english"))

        filtered_word=[]
        for w in word:
            if w not in stop_words or (w=='not'or w=='dont') :
                filtered_word.append(w)



        lemmatizer=WordNetLemmatizer()
        lemmatized_list=[]
        for w in filtered_word:
            if w!='not' or w!= 'dont' or w!="don't":
                lemmatized_list.append(lemmatizer.lemmatize(w))



        sentence=''.join(lemmatized_list)

        emotion=NRCLex(sentence)
   
        return(emotion.affect_frequencies)
    
    else:
        emotion=NRCLex(file)
        return (emotion.affect_frequencies)



def get_sentiment_value(emotion,file,word):
    output=sentiment(file,word)
    #print(output)

    trust_val=output.get(emotion)
    return(trust_val)



threshold=0.3
neg_threshold=0.3

positive_years=[]
year=2014
for i in range(0,6):
    file_name=str(year)+'_state_of_union.txt'
    if (get_sentiment_value('positive',file_name,'n')) >=threshold:
        positive_years.append(year)


        
    
    
    year+=1


year_positive_count=0
year=2014
dict={}
year_neg_count=[]
#
year_disgust_count=[]
year_sadness_count=[]
year_joy_count=[]
year_trust_count=[]
n_count=0
t_count=0
j_count=0
d_count=0
s_count=0
for i in range(0,6):
    x=open((str(year)+'_state_of_union.txt'),encoding='utf8')
    text=x.read()
    word_list=text.split()
    
    for w in word_list:
        if get_sentiment_value('positive',w,'word')>=threshold:
            year_positive_count+=1
        
        if get_sentiment_value('negative',w,'word') >= threshold:
            n_count+=1

        if get_sentiment_value('trust',w,'word') >=threshold:
            t_count+=1
       
        if get_sentiment_value('sadness',w,'word') >=threshold:
            s_count+=1
        
        if get_sentiment_value('disgust',w,'word') >=threshold:
            d_count+=1

        if get_sentiment_value('joy',w,'word') >=threshold:
            j_count+=1
        
    
    year_neg_count.append(n_count/len(word_list))
    year_joy_count.append(j_count/len(word_list))
    year_sadness_count.append(s_count/len(word_list))
    year_disgust_count.append(d_count/len(word_list))
    year_trust_count.append(t_count/len(word_list))
    
    dict[year]=year_positive_count/len(word_list)
    n_count=0
    j_count=0
    d_count=0
    t_count=0
    s_count=0
    year+=1
    year_positive_count=0
    #year_neg_count=0


#adding emotions



#print(dict)


#creating dataframe
import pandas as pd

year=2014
x=[]
#[1,0]
for i in range(0,6):
    if year in positive_years:
        x.append(1)
    else:
        x.append(0)
    
    year+=1



years=list(dict.keys())
wordcount=list(dict.values())


data=   {'Year':years,'Positive word count':wordcount,'Negative word count':year_neg_count,'Sad word count': year_sadness_count,'Joy word count': year_joy_count,'trust word count': year_trust_count,'disgust word count': year_disgust_count,'is positive':x}

train_data=pd.DataFrame(data)
#print(get_sentiment_value('negative','2018_state_of_union.txt','n'))
#print(get_sentiment_value('positive','2018_state_of_union.txt','n'))
#print(train_data)

test_dict={}
year_positive_coun=[]
year_neg_count.clear()
year_joy_count.clear()
year_disgust_count.clear()
year_sadness_count.clear()
year_trust_count.clear()



year=2020
year_positive_count=0
for i in range(0,5):
    x=open((str(year)+'_state_of_union.txt'),encoding='utf8')
    text=x.read()
    word_list=text.split()
    for w in word_list:
        if get_sentiment_value('positive',w,'word')>=threshold:
            year_positive_count+=1
        
        if get_sentiment_value('negative',w,'word') >= threshold:
            n_count+=1

        if get_sentiment_value('trust',w,'word') >=threshold:
            t_count+=1
       
        if get_sentiment_value('sadness',w,'word') >=threshold:
            s_count+=1
        
        if get_sentiment_value('disgust',w,'word') >=threshold:
            d_count+=1

        if get_sentiment_value('joy',w,'word') >=threshold:
            j_count+=1
        
    
    
    year_neg_count.append(n_count/len(word_list))
    year_joy_count.append(j_count/len(word_list))
    year_sadness_count.append(s_count/len(word_list))
    year_disgust_count.append(d_count/len(word_list))
    year_trust_count.append(t_count/len(word_list))
    year_positive_coun.append(year_positive_count/len(word_list))
    

    
    #test_dict[year]=year_positive_count
    n_count=0
    j_count=0
    d_count=0
    t_count=0
    s_count=0
    year_positive_count=0
    year+=1
    #year_positive_count=0


years=[2020,2021,2022,2023,2024]

new_data=   {'Year':years,'Positive word count':year_positive_coun,'Negative word count':year_neg_count,'Sad word count': year_sadness_count,'Joy word count': year_joy_count,'trust word count': year_trust_count,'disgust word count': year_disgust_count}

test_df=pd.DataFrame(new_data)
print('traindata', train_data)
print(test_df)
#print(test_df)


# regression

x=train_data.drop(columns='is positive')
y=train_data['is positive']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.1,random_state=42)

from sklearn.preprocessing import StandardScaler
print('x',x_test)
scaler=StandardScaler()

x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


from sklearn.linear_model import LogisticRegression

log_reg=LogisticRegression(random_state=42)
log_reg.fit(x_train,y_train)
#print(log_reg.predict(scaler.transform(test_df)))
print('predicted results',log_reg.predict(scaler.transform(test_df)))
#print(log_reg.predict(scaler.transform([393,152,60,96,217,13,1])))



positive_years.clear()
#print('actual results:')
year=2020
for i in range(0,5):
    file_name=str(year)+'_state_of_union.txt'
    #print(get_sentiment_value('positive',file_name,'n'))
    if (get_sentiment_value('positive',file_name,'n')) >=threshold:
        #print(get_sentiment_value('positive',file_name,'n'))
        positive_years.append(year)
    
    year+=1


year=2020
x=[]
#print(positive_years)
#[1,0]
for i in range(0,5):
    if year in positive_years:
        x.append(1)
    else:
        x.append(0)
    
    year+=1

print('actual results', x)