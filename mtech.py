import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, normaltest, mannwhitneyu
sns.set_theme(style="darkgrid")

st.write("""
# МТех 

Данное веб-приложение тестирует следующие гипотезы для ваших данных: 
         
1. Мужчины пропускают в течение года более 2 рабочих дней (work_days) поболезни значимо чаще женщин
         
2. Работники старше 35 лет (age) пропускают в течение года более 2 рабочих
дней (work_days) по болезни значимо чаще своих более молодых коллег
""")
st.write('---')

st.title('Каким образом происходит взаимодействие: ')

st.write('Ниже вы можете загрузить файл формата csv, который должен иметь следующий формат: ')

table_md = """
| id | Количество больничных дней | Возраст | Пол |
| --- | --- | --- | --- |
| int | int | int | 'Ж' / 'М' |

"""

st.markdown(table_md)

st.write(' ')
st.write('После загрузки вашего файла, вы можете задать интересующие вас параметры возраста и кол-ва пропущенных дней  - при помощи ползунка слева. Также можно выбрать уровень значимости, для которого будет происходить проверка гипотез. ')

st.write('После этого можно ознакомиться с визуализацией распределения ваших данных и результатами тестирования гипотез, пролистав вниз.')





st.title("Пожалуйста загрузите ваш csv файл.")

df = pd.read_csv('М.Тех_Данные_к_ТЗ_DS.csv', sep='\,', engine='python', header=None, encoding='cp1251').apply(lambda x: x.str.replace(r"\"","", regex=True))
rename_columns = {0: df[0][0], 1: df[1][0], 2: df[2][0]}
df.rename(rename_columns, axis=1, inplace=True)
df.drop(0, axis=0, inplace=True)
df = df.astype({'Количество больничных дней': 'int32', 'Возраст': 'int32'})
st.write(df)

uploaded_file = st.file_uploader("Выбрать файл")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep='\,', engine='python', header=None, encoding='cp1251').apply(lambda x: x.str.replace(r"\"","", regex=True))
    rename_columns = {0: df[0][0], 1: df[1][0], 2: df[2][0]}
    df.rename(rename_columns, axis=1, inplace=True)
    df.drop(0, axis=0, inplace=True)
    df = df.astype({'Количество больничных дней': 'int32', 'Возраст': 'int32'})
    st.write(df)

st.title("Гипотеза 1.")

st.write('Мужчины пропускают в течение года более 2 рабочих дней (work_days) по болезни значимо чаще женщин.')

fig1 = plt.figure(figsize=(9, 5))
sns.histplot(data=df, x="Пол")
plt.title('Количество мужчин и женщин в нашем датасете')
st.pyplot(fig1)

fig2 = plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="Количество больничных дней", hue="Пол")
plt.title('Распределение между полом и количеством пропущенных по болезни дней')
st.pyplot(fig2)

fig3 = plt.figure(figsize=(9, 5))
sns.histplot(data=df, x="Количество больничных дней", hue="Пол", element="poly", kde=True)
st.pyplot(fig3)

class mean_test_between_men_and_women:

    def __init__(self, df, alpha=0.05, work_days=2):
        # :df: input [DataFrame] 
        # :alpha: statistical significance level [int]
        # :work_days: number of missed working days we want to explore [int]
        self.df = df
        self.alpha = alpha
        self.work_days = work_days

    def df_cutting(self):

        #нам интересно число пропущенных дней > work_days
        df_cut = self.df[self.df['Количество больничных дней'] > self.work_days]

        #создадим две подвыборки с пропущенными дням, соответствующие мужчинам и женщинам
        men = df_cut[df_cut['Пол'] == 'М']['Количество больничных дней'].values
        women = df_cut[df_cut['Пол'] == 'Ж']['Количество больничных дней'].values

        return men, women

    def t_test(self, men, women):
        #t-test, мы можем считать наши распределения независимыми, поэтому используем ttest_ind из scipy библотеки 
        stat, p = ttest_ind(men, women, alternative='greater') #alternative='greater' означает, что согласно альтернативной гипотезы 𝜇_men > 𝜇_women
        if p <= self.alpha: 
            st.markdown(f'T-test: Нулевая гипотеза о том, что 𝜇_women ≥ 𝜇_men отвергается, т.к. p-value для t-test {round(p, 3)} < {self.alpha} - уровня значимости.')
            st.markdown('\nМы принимаем гипотезу о том, что мужчины пропускают в течение года более ' + str(self.work_days) + ' рабочих дней (work_days) по болезни значимо чаще женщин.\n')
            return ' '
        else: 
            st.markdown(f'T-test: Нулевая гипотеза о том, что 𝜇_women ≥ 𝜇_men не может быть отвергнута, т.к. p-value для t-test {round(p, 3)} > {self.alpha} - уровня значимости.')
            st.markdown('\nМы отвергаем гипотезу о том, что мужчины пропускают в течение года более ' + str(self.work_days) + ' рабочих дней (work_days) по болезни значимо чаще женщин.\n')
            return ' '

    def mannwhitneyu_test(self, men, women):
        #Mann-Whitney U test:
        stat, p = mannwhitneyu(men, women, alternative='greater') #alternative='greater' означает, что согласно альтернативной гипотезы 𝜇_men > 𝜇_women
        if p <= self.alpha: 
            st.markdown(f'Mann-Whitney U test: Нулевая гипотеза о том, что 𝜇_women ≥ 𝜇_men отвергается, т.к. p-value для Mann-Whitney U test {round(p, 3)} < {self.alpha} - уровня значимости.')
            st.markdown('\nМы принимаем гипотезу о том, что мужчины пропускают в течение года более ' + str(self.work_days) + ' рабочих дней (work_days) по болезни значимо чаще женщин.\n')
            return ' '
        else: 
            st.markdown(f'Mann-Whitney U test: Нулевая гипотеза о том, что 𝜇_women ≥ 𝜇_men не может быть отвергнута, т.к. p-value для Mann-Whitney U test {round(p, 3)} > {self.alpha} - уровня значимости.')
            st.markdown('\nМы отвергаем гипотезу о том, что мужчины пропускают в течение года более ' + str(self.work_days) + ' рабочих дней (work_days) по болезни значимо чаще женщин.\n')
            return ' '

    def test(self):  
         #тест на нормальность
         men, women = self.df_cutting()
         if len(men) < 8 or len(women) < 8: 
            st.markdown('Невозможно провести тесты, так как размеры подвыборки малы: n<8, добавьте данных в датасет или же уменьшите параметр work_days.')
            return 
         
         stat_norm_men, p_norm_men = normaltest(men)
         stat_norm_women, p_norm_women = normaltest(women)
         #также проверяем размер наших подвыборок, в случае больших подвыборок - мы можем проводить t-test
         if (p_norm_men <= self.alpha or p_norm_women <= self.alpha) and (len(men) < 40 or len(women) < 40):
             st.markdown(f'Мы не можем провести t-test так как мы не можем принять гипотезу о нормальности распределенных данных или же у нас маленький датасет, результаты теста на нормальность: \nдля мужчин p-value = {round(p_norm_men, 3)} \nдля женщин p-value = {round(p_norm_women, 3)} \nна уровне значимости alpha = {self.alpha}.')
             st.markdown('\nНо можем провести непараметрический тест Манна — Уитни:')
             self.mannwhitneyu_test(men, women)
         else: 
             self.t_test(men, women), self.mannwhitneyu_test(men, women)

         return 


def user_input_features():
    age = st.sidebar.slider('Возраст', df['Возраст'].min(), df['Возраст'].max(), int(df['Возраст'].mean()), step=1)
    work_days = st.sidebar.slider('Количество пропущенных рабочих дней', df['Количество больничных дней'].min(), df['Количество больничных дней'].max(), int(df['Количество больничных дней'].mean()), step=1)
    alpha = st.sidebar.slider('Уровень значимости 𝛼', 0.01, 0.1, 0.05)

    return age, work_days, alpha


age, work_days, alpha = user_input_features()

st.title("Результаты тестирования гипотезы 1.")

testing1 = mean_test_between_men_and_women(df, alpha, work_days)
testing1.test()

st.write('Результаты бутстрепа:')


class bootstrap_for_men_and_women():

    def __init__(self, df, alpha=0.05, k=10000):
        # :df: input [DataFrame] 
        # :k: number of subsamples [int]
        # :alpha: statistical significance level [int]
        self.df = df
        self.k = k
        self.alpha = alpha

    def men_and_women(self):
        #создадим две подвыборки, соответствующие мужчинам и женщинам
        men = self.df[self.df['Пол'] == 'М']['Количество больничных дней'].values
        women = self.df[self.df['Пол'] == 'Ж']['Количество больничных дней'].values

        return men, women

    def sampling(self, data): 
        #генерим выборки для бутстрепа
        index = np.random.randint(0, len(data), (self.k, len(data)))
        samples = data[index]

        return samples

    def conf_intervals(self, mean):
        # функция для интервальной оценки
        # :mean: list of mean values for our subsamples [list]
        boundaries = np.percentile(mean, [100 * self.alpha / 2., 100 * (1 - self.alpha / 2.)])

        return boundaries
    
    def generation(self): 
        men, women = self.men_and_women()
        men_means = [np.mean(sample) for sample in self.sampling(men)]
        women_means = [np.mean(sample) for sample in self.sampling(women)]

        return men_means, women_means

    def intervals(self): 
        men_means, women_means = self.generation()
        conf_int_men = self.conf_intervals(men_means)
        conf_int_women = self.conf_intervals(women_means)
        st.markdown(f'С {int(100*(1-self.alpha))}% вероятностью мужчина пропускает кол-во рабочих дней в диапазоне от {round(conf_int_men[0], 2)} до {round(conf_int_men[1], 2)}.')
        st.markdown(f'С {int(100*(1-self.alpha))}% вероятностью женщина пропускает кол-во рабочих дней в диапазоне от {round(conf_int_women[0], 2)} до {round(conf_int_women[1], 2)}.   ')


bootstrap1 = bootstrap_for_men_and_women(df, alpha)
bootstrap1.intervals()

st.title("Гипотеза 2.")

st.write('Работники старше ' + str(age) + ' лет (age) пропускают в течение года более 2 рабочих дней (work_days) по болезни значимо чаще своих более молодых коллег.')

fig4 = plt.figure(figsize=(9, 5))
df['Ранжировка'] = df['Возраст']
df['Ранжировка'] = df['Ранжировка'].map(lambda x: 'До ' + str(age) if x < age else 'Старшe ' + str(age))
sns.histplot(data=df, x="Ранжировка")
plt.title('Количество пропущенных дней работниками до ' + str(age) + ' и старше')
st.pyplot(fig4)

fig5 = plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="Количество больничных дней", hue="Ранжировка")
plt.title('Распределение пропущенных по болезни дней работниками до ' + str(age) + ' и старше')
st.pyplot(fig5)

fig6 = plt.figure(figsize=(9, 5))
sns.histplot(data=df, x="Количество больничных дней", hue="Ранжировка", element="poly", kde=True)
st.pyplot(fig6)

class mean_test_between_young_and_aged:

    def __init__(self, df, alpha=0.05, work_days=2):
        # :df: input [DataFrame] 
        # :alpha: statistical significance level [int]
        # :work_days: number of missed working days we want to explore [int]
        self.df = df
        self.alpha = alpha
        self.work_days = work_days

    def df_cutting(self):

        #нам интересно число пропущенных дней > work_days
        df_cut = self.df[self.df['Количество больничных дней'] > self.work_days]

        #создадим две подвыборки с пропущенными дням, соответствующие мужчинам и женщинам
        aged = df_cut[df_cut['Ранжировка'] == 'До ' + str(age)]['Количество больничных дней'].values
        young = df_cut[df_cut['Ранжировка'] == 'Старшe ' + str(age)]['Количество больничных дней'].values

        return aged, young

    def t_test(self, aged, young):
        #t-test, мы можем считать наши распределения независимыми, поэтому используем ttest_ind из scipy библотеки 
        stat, p = ttest_ind(aged, young, alternative='greater') #alternative='greater' означает, что согласно альтернативной гипотезы 𝜇_aged > 𝜇_young
        if p <= self.alpha: 
            st.markdown(f'T-test: Нулевая гипотеза о том, что 𝜇_young ≥ 𝜇_aged отвергается, т.к. p-value для t-test {round(p, 3)} < {self.alpha} - уровня значимости.')
            st.markdown('\nМы принимаем гипотезу о том, что взрослые работники пропускают в течение года более 2 рабочих дней (work_days) по болезни значимо чаще молодых.\n')
        else: 
            st.markdown(f'T-test: Нулевая гипотеза о том, что 𝜇_young ≥ 𝜇_aged не может быть отвергнута, т.к. p-value для t-test {round(p, 3)} > {self.alpha} - уровня значимости.')
            st.markdown('\nМы отвергаем гипотезу о том, что взрослые работники пропускают в течение года более 2 рабочих дней (work_days) по болезни значимо чаще молодых.\n')

    def mannwhitneyu_test(self, aged, young):
        #Mann-Whitney U test:
        stat, p = mannwhitneyu(aged, young, alternative='greater') #alternative='greater' означает, что согласно альтернативной гипотезы 𝜇_aged > 𝜇_young
        if p <= self.alpha: 
            st.markdown(f'Mann-Whitney U test: Нулевая гипотеза о том, что 𝜇_young ≥ 𝜇_aged отвергается, т.к. p-value для Mann-Whitney U test {round(p, 3)} < {self.alpha} - уровня значимости.')
            st.markdown('\nМы принимаем гипотезу о том, что взрослые работники пропускают в течение года более 2 рабочих дней (work_days) по болезни значимо чаще молодых.\n')
        else: 
            st.markdown(f'Mann-Whitney U test: Нулевая гипотеза о том, что 𝜇_young ≥ 𝜇_aged не может быть отвергнута, т.к. p-value для Mann-Whitney U test {round(p, 3)} > {self.alpha} - уровня значимости.')
            st.markdown('\nМы отвергаем гипотезу о том, что взросыле работники пропускают в течение года более 2 рабочих дней (work_days) по болезни значимо чаще молодых.\n')

    def test(self):  
         #тест на нормальность
         aged, young = self.df_cutting()
         if len(aged) < 8 or len(young) < 8: 
            st.markdown('Невозможно провести тесты, так как размеры подвыборки малы: n<8, добавьте данных в датасет или же уменьшите параметр work_days.')
            return
         stat_norm_aged, p_norm_aged = normaltest(aged)
         stat_norm_young, p_norm_young = normaltest(young)
         #также проверяем размер наших подвыборок, в случае больших подвыборок - мы можем проводить t-test
         if (p_norm_aged <= self.alpha or p_norm_young <= self.alpha) and (len(aged) < 40 or len(young) < 40):
             st.markdown(f'Мы не можем провести t-test так как мы не можем принять гипотезу о нормальности распределенных данных или же у нас маленький датасет, результаты теста на нормальность: \nдля взрослых работников p-value = {round(p_norm_aged, 3)} \nдля молодых работников p-value = {round(p_norm_young, 3)} \nна уровне значимости alpha = {self.alpha}.')
             st.markdown('\nНо можем провести непараметрический тест Манна — Уитни:')
             self.mannwhitneyu_test(aged, young)
         else: 
             self.t_test(aged, young), self.mannwhitneyu_test(aged, young)
         return 

st.title("Результаты тестирования гипотезы 2.")

testing2 = mean_test_between_young_and_aged(df, alpha, work_days)
testing2.test()


st.write('Результаты бутстрепа:')

class bootstrap_for_young_and_aged():

    def __init__(self, df, alpha=0.05, k=10000):
        # :df: input [DataFrame] 
        # :k: number of subsamples [int]
        # :alpha: statistical significance level [int]
        self.df = df
        self.k = k
        self.alpha = alpha

    def aged_and_young(self):
        #создадим две подвыборки, соответствующие мужчинам и женщинам
        aged = df[df['Ранжировка'] == 'До ' + str(age)]['Количество больничных дней'].values
        young = df[df['Ранжировка'] == 'Старшe ' + str(age)]['Количество больничных дней'].values

        return aged, young

    def sampling(self, data): 
        #генерим выборки для бутстрепа
        index = np.random.randint(0, len(data), (self.k, len(data)))
        samples = data[index]

        return samples

    def conf_intervals(self, mean):
        # функция для интервальной оценки
        # :mean: list of mean values for our subsamples [list]
        boundaries = np.percentile(mean, [100 * self.alpha / 2., 100 * (1 - self.alpha / 2.)])

        return boundaries
    
    def generation(self): 
        aged, young = self.aged_and_young()
        aged_means = [np.mean(sample) for sample in self.sampling(aged)]
        young_means = [np.mean(sample) for sample in self.sampling(young)]

        return aged_means, young_means

    def intervals(self): 
        aged_means, young_means = self.generation()
        conf_int_aged = self.conf_intervals(aged_means)
        conf_int_young = self.conf_intervals(young_means)
        st.markdown(f'С {int(100*(1-self.alpha))}% вероятностью взрослый работник пропускает кол-во рабочих дней в диапазоне от {round(conf_int_aged[0], 2)} до {round(conf_int_aged[1], 2)}.')
        st.markdown(f'С {int(100*(1-self.alpha))}% вероятностью молодой работник пропускает кол-во рабочих дней в диапазоне от {round(conf_int_young[0], 2)} до {round(conf_int_young[1], 2)}.')


bootstrap2 = bootstrap_for_young_and_aged(df, alpha)
bootstrap2.intervals()


