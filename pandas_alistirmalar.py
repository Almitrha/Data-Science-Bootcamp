

##################################################
# Pandas Alıştırmalar
##################################################

import numpy as np
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#########################################
# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
#########################################
titanic_data = sns.load_dataset("titanic")

#########################################
# Görev 2: Yukarıda tanımlanan Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
#########################################
titanic_data.head()

titanic_data.value_counts("sex")


#########################################
# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
#########################################
titanic_data.nunique()


#########################################
# Görev 4: pclass değişkeninin unique değerleri bulunuz.
#########################################

titanic_data["pclass"].unique()


#########################################
# Görev 5:  pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
#########################################

len(titanic_data["pclass"].unique())
len(titanic_data["parch"].unique())
#########################################
# Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz. Tekrar tipini kontrol ediniz.
#########################################

titanic_data["embarked"].dtype

titanic_data["embarked"]=titanic_data["embarked"].astype("category")

#########################################
# Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.
#########################################

embarked_c_passengers = titanic_data.loc[titanic_data["embarked"] == "C"]


#########################################
# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
#########################################

embarked_nons_passengers = titanic_data.loc[titanic_data["embarked"] != "S"]

#########################################
# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
#########################################

age_below_30_and_female = titanic_data.loc[(titanic_data["age"] < 30) & (titanic_data["sex"] == "female")]

#########################################
# Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
#########################################

rich_oldies =  titanic_data.loc[(titanic_data["fare"] > 500) & (titanic_data["age"] > 70)]

#########################################
# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
#########################################

titanic_data.isnull().sum()


#########################################
# Görev 12: who değişkenini dataframe'den düşürün.
#########################################

titanic_data.drop("who", axis=1)

#########################################
# Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
#########################################

titanic_data["deck"].fillna(titanic_data["deck"].mode()[0], inplace= True)



#########################################
# Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurun.
#########################################

titanic_data["age"].fillna(titanic_data["age"].median(), inplace=True)

#########################################
# Görev 15: survived değişkeninin Pclass ve Cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
#########################################
titanic_data.pivot_table(values="survived", index=["pclass", "sex"], aggfunc=["sum", "count", "mean"])

#########################################
# Görev 16:  30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazınız.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)
#########################################
def age_flag(age):
    if age < 30:
        return 1
    else:
        return 0

titanic_data["age_flag"]=titanic_data["age"].apply(lambda x: age_flag(x))
titanic_data.head()

#########################################
# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
#########################################

tips = sns.load_dataset("tips")


#########################################
# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill  değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################

tips.pivot_table(values="total_bill", index="time", aggfunc=["sum","min", "max", "mean"])

#########################################
# Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
########################################
tips.head()
tips.pivot_table(values="total_bill", index= ["time", "day"], aggfunc=["min", "max", "sum", "mean"])
#########################################
# Görev 20:Lunch zamanına ve kadın müşterilere ait total_bill ve tip  değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
#########################################

result = tips[(tips['time'] == 'Lunch') & (tips['sex'] == 'Female')].groupby('day').agg({'total_bill': ['sum', 'min', 'max', 'mean'], 'tip': ['sum', 'min', 'max', 'mean']})

#########################################
# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?
#########################################

orders = tips[(tips["size"] < 3) & (tips["total_bill"] > 10)]
orders["total_bill"].mean()

#########################################
# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturun. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
#########################################

tips["total_bill_tip_sum"] = tips["total_bill"] +  tips["tip"]
tips.head()
#########################################
# Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
#########################################

sorted_values = tips.sort_values("total_bill_tip_sum", ascending=False)

new_df = sorted_values.head(30)