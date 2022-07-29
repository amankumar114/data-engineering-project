#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pyspark


# In[18]:


from pyspark.sql import SparkSession
from pyspark.sql import Row
spark = SparkSession.builder.appName("ML model").getOrCreate()

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation  import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, VectorIndexer
from pyspark.sql import functions as f


# In[19]:


#SparkSession -entrypoint to spark - creates Session for the user 

from pyspark.sql import SparkSession 
from pyspark.sql import Row
spark = SparkSession.builder.appName("CP2G1").getOrCreate()


sc = spark.sparkContext


# In[4]:



dev_demog_df = spark.read.csv("/user/unextbdh22id010/Dev_Demog .csv",header = True)

dev_demog_df.show(5)


# In[5]:


dev_demog_df.printSchema()


# In[6]:


#Typecast
from pyspark.sql.types import IntegerType,FloatType,StringType

df_a=dev_demog_df    .withColumn("ever60_24m_StrictFlag",
               dev_demog_df["ever60_24m_StrictFlag"]
               .cast(FloatType())) \
   .withColumn("Max_dependent",
               dev_demog_df["Max_dependent"]
               .cast(FloatType())) \
   .withColumn("LTV",
               dev_demog_df["LTV"]
               .cast(FloatType())) \
   .withColumn("Doc_form_16",
               dev_demog_df["Doc_form_16"]
               .cast(FloatType())) \
   .withColumn("income_max",
               dev_demog_df["income_max"]
               .cast(FloatType())) \
   .withColumn("income_sum",
               dev_demog_df["income_sum"]
               .cast(FloatType())) \
   .withColumn("income_min",
               dev_demog_df["income_min"]
               .cast(FloatType())) \
    .withColumn("edu_max",
               dev_demog_df["edu_max"]
               .cast(FloatType())) \
    .withColumn("edu_min",
               dev_demog_df["edu_min"]
               .cast(FloatType())) \
    .withColumn("age_max",
               dev_demog_df["age_max"]
               .cast(FloatType())) \
    .withColumn("emi_income_sum",
               dev_demog_df["emi_income_sum"]
               .cast(FloatType())) \
     .withColumn("Dummy_application_id",
               dev_demog_df["Dummy_application_id"]
               .cast(FloatType())) \
    .withColumn("REQUESTED_TENURE",
               dev_demog_df["REQUESTED_TENURE"]
               .cast(FloatType())) \
    .withColumn("PROPERTY_INSURANCE_AMT",
               dev_demog_df["PROPERTY_INSURANCE_AMT"]
               .cast(FloatType())) \
    .withColumn("City_tier",
               dev_demog_df["City_tier"]
               .cast(FloatType())) \
    .withColumn("emi_income_max",
               dev_demog_df["emi_income_max"]
               .cast(FloatType())) \
    .withColumn("FOIR2",
                dev_demog_df["FOIR2"]
                .cast(FloatType())) \
    .withColumn("age_min",
                dev_demog_df["age_min"]
                .cast(FloatType())) \
    .withColumn("cnt_coapplicant",
                dev_demog_df["cnt_coapplicant"]
                .cast(FloatType())) \
df_a.printSchema()


# In[7]:


pandasDF = df_a.toPandas()
print(pandasDF)


# In[8]:


pandasDF.isnull().sum()


# In[9]:


pandasDF['emi_income_sum'].fillna(pandasDF['emi_income_sum'].mean(),inplace=True)
pandasDF['emi_income_max'].fillna(pandasDF['emi_income_max'].mean(),inplace=True)
pandasDF['LTV'].fillna(pandasDF['LTV'].mean(),inplace=True)
pandasDF['FOIR2'].fillna(pandasDF['FOIR2'].mean(),inplace=True)
pandasDF['income_max'].fillna(pandasDF['income_max'].mean(),inplace=True)
pandasDF['income_sum'].fillna(pandasDF['income_sum'].mean(),inplace=True)
pandasDF['income_min'].fillna(pandasDF['income_min'].mean(),inplace=True)
pandasDF['City_tier'].fillna(pandasDF['City_tier'].mode([0]),inplace=True)
pandasDF['NATURE_OF_ORGANISATION'].fillna('unknown',inplace=True)
pandasDF['OCCUPATION_TYPE'].fillna('unknown',inplace=True)


# In[10]:


pandasDF.isnull().sum()


# In[12]:


fin_df = pandasDF.drop(['NATURE_OF_ORGANISATION','Max_dependent','SALARIED_SELF','emi_income_sum','edu_max','age_min','cnt_coapplicant'], axis=1)


# In[13]:


fin_df.columns


# In[14]:


fin_df.isnull().sum()


# In[15]:


DemogDF=spark.createDataFrame(fin_df)
DemogDF.printSchema()


# In[20]:


numeric_features = [t[0] for t in DemogDF.dtypes if t[1] == 'double']
DemogDF.select(numeric_features).describe().toPandas().transpose()


# In[23]:


output = assembler.transform(DemogDF)
output.head()


# In[22]:


assembler = VectorAssembler(inputCols=['emi_income_max','LTV','FOIR2','Doc_form_16','income_max'],
                            outputCol='features')


# In[24]:



training_data = output.select('features','ever60_24m_StrictFlag')
training_data.head(5)


# In[25]:


train_set,test_set =training_data.randomSplit([0.7,0.3])


# In[26]:


train_set.show(3)


# In[27]:



test_set.show(3)


# In[28]:


classifier = LogisticRegression(labelCol = 'ever60_24m_StrictFlag')


# In[29]:


reg_model = classifier.fit(train_set)


# In[30]:


predict_data = reg_model.transform(test_set)


# In[31]:


evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='ever60_24m_StrictFlag')

AUC = evaluator.evaluate(predict_data)


# In[32]:


accuracy = (reg_model,predict_data)
accuracy


# In[33]:


pred = predict_data.select('prediction')
out =  predict_data.select('ever60_24m_StrictFlag')


# In[34]:


pred_pd = pred.toPandas()
prednp = pred_pd.to_numpy()
print(prednp)
out_pd = out.toPandas()
outnp = out_pd.to_numpy()
print(outnp)


# In[35]:


from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(prednp,outnp )
roc_auc = auc(fpr, tpr)
import matplotlib.pyplot as plt

plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




