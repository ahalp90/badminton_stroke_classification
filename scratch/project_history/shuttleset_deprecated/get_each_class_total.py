"""DEPRECATED (2026-04-08) — Partially superseded by pipeline/ modules.

Aggregates per-class stroke counts from ShuttleSet CSV annotations into
an Excel workbook, with separate sheets for Top/Bottom players.

Superseded by:
    pipeline/player_mapping.py  — A/B -> Top/Bottom mapping and shot
        collection (replaces inline set-3 logic at lines 38-43).
    pipeline/config.py          — centralised paths and stroke types.

To align with the current codebase (if ever needed):
    1. Replace hardcoded paths ('set/match.csv', 'class_total.xlsx',
       etc.) with ``from pipeline.config import SET_INFO_DIR``
    2. Replace inline set-3 XOR logic (lines 38-43) with a second call
       to ``map_players()`` (already imported but not used for set 3)
    3. Replace Chinese error messages (line 67) with English
"""
import sys
from pathlib import Path

# Allow importing pipeline when running from ShuttleSet/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from pipeline.player_mapping import map_players, find_set3_switch_rally


def get_one_competition_result(folder_path: Path, first_A_is_top: bool):
    '''A is the winner and B is the loser.'''
    set_count = len(list(folder_path.glob('*.csv')))
    sets_collected_ls = []

    # For set1 and set2
    for i in range(1, 3):
        df = pd.read_csv(folder_path/f'set{i}.csv')
        df = df[['player', 'type']]
        df = map_players(df, first_A_is_top, i)
        type_count = df.groupby(['player', 'type']).size()
        sets_collected_ls.append(type_count)

    # Handle set 3 court switch at 11 points
    if set_count == 3:
        df = pd.read_csv(folder_path/'set3.csv')
        df = df[['roundscore_A', 'roundscore_B', 'player', 'type', 'rally']]

        i_split = find_set3_switch_rally(df)
        df_1 = df.iloc[:i_split].copy()
        df_2 = df.iloc[i_split:].copy()

        df_1 = df_1[['player', 'type']]
        df_2 = df_2[['player', 'type']]

        # Before switch: same as set 1 mapping
        if first_A_is_top:
            df_1['player'] = np.where(df_1['player'] == 'A', 'Top', 'Bottom')
            df_2['player'] = np.where(df_2['player'] == 'B', 'Top', 'Bottom')
        else:
            df_1['player'] = np.where(df_1['player'] == 'B', 'Top', 'Bottom')
            df_2['player'] = np.where(df_2['player'] == 'A', 'Top', 'Bottom')

        count_1 = df_1.groupby(['player', 'type']).size()
        count_2 = df_2.groupby(['player', 'type']).size()
        sets_collected_ls += [count_1, count_2]

    df = pd.concat(sets_collected_ls, axis=1)
    df = df.fillna(0)
    result = df.astype(int).sum(axis=1).sort_index()
    return result


def update_dataframes(df_dic: dict[str, pd.DataFrame], competition_name: str, result: pd.Series):
    '''Update dataframes.'''
    df_top = df_dic['Top Player']

    row = df_top[df_top['Video Name'] == competition_name].index
    for player in ['Top', 'Bottom']:
        cur_sheet = f'{player} Player'
        for stroke_name, stroke_count in result[player].items():
            col = np.argmax(df_dic[cur_sheet].columns == stroke_name)
            if col != 0:  # dataset 有未知球種
                df_dic[cur_sheet].iloc[row, col] = stroke_count
            else:
                raise Exception(f'球種 {stroke_name} 沒有統計到')


if __name__ == "__main__":
    match_csv_df = pd.read_csv('set/match.csv')[['video', 'downcourt']].set_index('video')
    match_ls = [p for p in Path("set").glob('*') if p.is_dir()]
    df_dic = pd.read_excel("class_total.xlsx", sheet_name=['Top Player', 'Bottom Player'])
    
    # 歸零
    df_dic['Top Player'].iloc[:, 2:] = 0
    df_dic['Bottom Player'].iloc[:, 2:] = 0
    
    for competition in match_ls:
        # if competition.name == 'CHEN_Long_CHOU_Tien_Chen_World_Tour_Finals_Group_Stage':
        #     os.system('pause')
        first_A_is_top = bool(match_csv_df.loc[competition.name, 'downcourt'])  # downcourt 是 1 代表 B 為 Bottom player
        result = get_one_competition_result(competition, first_A_is_top)
        # print(competition.name)
        # print(result.sum())
        update_dataframes(df_dic, competition.name, result)
    
    with pd.ExcelWriter('class_total_gen.xlsx') as writer:
        df_dic['Top Player'].to_excel(writer, sheet_name='Top Player', index=False)
        df_dic['Bottom Player'].to_excel(writer, sheet_name='Bottom Player', index=False)
