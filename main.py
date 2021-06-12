import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import random

st.title('Прогнозирование цены футболистов')

st.markdown('Целью этого проекта было предсказание цен на молодых атакующих футболистов после следующего сезона. '
         'Основу обучающей выборки составили около 400 игроков из топ-5 Европейских футбольных лиг и '
         'Российского чемпионата. Цены всех футболистов взяты с сайта [transfermarkt.com ](https://transfermarkt.com ). '
         'Полное описание всех этапов построения предсказательной модели, осованной на машинном обучении (использовался градиентный бустинг), '
         'вы найдете в соответсвтующем Jupiter Notebook. Это веб-приложение же визуализирует полученные результаты. '
            'Выберите интересующего вас игрока и узнайте как изменится его цена в через год!')
data = pd.read_csv('final_predictions.csv')
data.price_predicted = data.price_predicted.apply(round)

def change_name(s):
    res = ''
    for a in s.split():
        res += a[0].capitalize()+a[1:]
        res += ' '
    return res[:-1]

data.player_name = data.player_name.apply(change_name)

leagues = data.League.unique()

teams_by_league = {}
for i in leagues:
    teams_by_league[i] = data[data['League'] == i]['team_title'].unique()

player_by_team = {}
for j in data.team_title.unique():
    player_by_team[j] = data[data['team_title'] == j]['player_name'].unique()

col1, col2 = st.beta_columns(2)

league1 = col1.selectbox('First Player', leagues, index= 1)
league2 = col2.selectbox('Second Player', leagues, index = 1)
team1 = col1.selectbox('Pick club', teams_by_league[league1], index= 1)
team2 = col2.selectbox('Pick club', teams_by_league[league2], index= 2)
if len(player_by_team[team1]) == 1:
    player1 = player_by_team[team1][0]
    col1.markdown(player1 + ' is the only player under 23')
    col1.markdown('in ' + team1)
else:
    player1 = col1.selectbox('Pick player', player_by_team[team1], index= 0)

if len(player_by_team[team2]) == 1:
    player2 = player_by_team[team2][0]
    col2.markdown(player2 + ' is the only player under 23')
    col2.markdown('in ' + team2)
else:
    player2 = col2.selectbox('Pick player', player_by_team[team2], index= 1)

data['goals_90'] = (data['goals_season'] * 90) / data['time']
data['assists_90'] = (data['assists_season'] * 90) / data['time']

categories = ['xGBuildup 90', 'assists', 'xG 90', 'xA 90', 'goals']
stats1 = list(data[data['player_name'] == player1][['xGBuildup_90','assists_90', 'npxG_season_90', 'xA_season_90', 'goals_90']].values[0])

stats2 = list(data[data['player_name'] == player2][['xGBuildup_90', 'assists_90', 'npxG_season_90',
                                                                                   'xA_season_90', 'goals_90']].values[0])
angles = list(np.linspace(0, 2 * np.pi, len(categories), endpoint=False))
angles += angles[:1]
stats1 += stats1[:1]
stats2 += stats2[:1]

fig = plt.figure()
ax = plt.subplot(111, polar=True)

ax.set_theta_offset(np.pi / 2)    ###FROM: https://www.python-graph-gallery.com/391-radar-chart-with-several-individuals
ax.set_theta_direction(-1)

plt.xticks(angles[:-1], categories)

ax.set_rlabel_position(0)

max_stat = np.max(stats1 + stats2)   ###END FROM

plt.yticks([max_stat * 0.25, max_stat * 0.5,
            max_stat * 0.75], [str(round(max_stat * 0.25, 2)), str(round(max_stat * 0.5, 2)),
                               str(round(max_stat * 0.75, 2))], color="grey", size=7)
plt.ylim(0, max_stat)

ax.plot(angles, stats1, linewidth=1, linestyle='solid', label= player1)
ax.fill(angles, stats1, 'b', alpha=0.1)

ax.plot(angles, stats2, linewidth=1, linestyle='solid', label= player2)
ax.fill(angles, stats2, 'r', alpha=0.1)

plt.legend(loc='lower left', bbox_to_anchor=(0.9, 0.9))

st.pyplot(fig)

player1_price = data[data['player_name'] == player1]['price_after_season']
player2_price = data[data['player_name'] == player2]['price_after_season']

player1_price_proj = data[data['player_name'] == player1]['price_predicted']
player2_price_proj = data[data['player_name'] == player2]['price_predicted']

col11, col22 = st.beta_columns(2)
col11.write(player1 + ' сейчас стоит ' + str(int(player1_price)) + ' т. €')
col11.write('Его ожидаемая цена через год - ' + str(int(player1_price_proj)) + ' т. €')

col22.write(player2 + ' сейчас стоит ' + str(int(player2_price)) + ' т. €')
col22.write('Его ожидаемая цена через год - ' + str(int(player2_price_proj)) + ' т. €')

st.write("Голы и ассисты приведены из расчета на 90 минут. С продвинутой статистикой можно ознакомиться"
         "на сайте [understat.com](https://understat.com)")

st.write("Не забывайте, что это мини-приложение - лишь инструмент визуализации результатов предсказательной модели. "
         "Основная часть работы находится в приложенном Jupiter Notebook")
#python  -m streamlit run main.py

