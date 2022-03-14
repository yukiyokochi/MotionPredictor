import pandas as pd
import glob, re, os

width_of_csv = 50


path = 'csv/*.csv'
all_csv = glob.glob(path)
for csv in all_csv:
    df = pd.read_csv(csv)
    w = width_of_csv
    csv_num = re.sub(r"\D", "", csv)
    if len(df) < w:
        continue
    else:
        for start in range(len(df) - w + 1):
            new_df = df.loc[start : start + w -1]
            dir = 'csv/devided_' + str(csv)
            if not os.path.isdir(dir):
                os.makedirs(dir)
            new_df.to_csv('csv/devided_' +  str(csv) + '/' + str(csv_num)+ '_' + str(start) + '_width' + str(w) + '.csv')
