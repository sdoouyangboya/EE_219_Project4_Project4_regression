import pandas as pd
import pycountry_convert as pc
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_datasets():
    bike_df = pd.read_csv('input/Bike-Sharing-Dataset/day.csv')
    suicide_df = pd.read_csv('input/master.csv')
    suicide_df[' gdp_for_year ($) '] = suicide_df[' gdp_for_year ($) '].apply(lambda x: float(x.split()[0].replace(',', '')))
    video_df = pd.read_csv('input/online_video_dataset/transcoding_mesurment.tsv',sep='\t')
    return bike_df, suicide_df, video_df


def q4_helper():
    bike_df = pd.read_csv('input/Bike-Sharing-Dataset/day.csv', usecols=['yr', 'mnth', 'cnt'])
    year_column = bike_df['yr']
    max_year = year_column.max()
    month_column = bike_df['mnth']
    smallest_month = month_column.min()
    max_month = month_column.max()

    count_list = {}
    for i in range(max_year + 1):
        for j in range(smallest_month, max_month + 1, 3):
            bike_df_month = bike_df[(bike_df['mnth'] == j) & (bike_df['yr'] == i)]
            days_list = []
            for _, rows in bike_df_month.iterrows():
                days_list.append(rows['cnt'])
            count_list[i,j] = days_list

    return max_year, smallest_month, max_month, count_list

def q5_helper():
    suicide_df = pd.read_csv('input/master.csv')
    countries = suicide_df.groupby(['country'])
    ten_largest = (countries['year'].max()-countries['year'].min()).sort_values(ascending=False)[0:10].index.tolist()
    return suicide_df, ten_largest
    # agg_df = pd.DataFrame(countries['year'].max()).rename(columns={'year':'maxyear'}).merge( pd.DataFrame(countries['year'].min()).rename(columns={"year":"minyear"}),left_index=True,right_index=True)
    # agg_df['diff'] = agg_df['maxyear']-agg_df['minyear']


def q6_helper():
    video_df = pd.read_csv('input/online_video_dataset/transcoding_mesurment.tsv',sep='\t')
    video_df_utime = pd.DataFrame(video_df['utime'])
    return video_df_utime

def country_to_continent(country_name):
    if country_name == 'Republic of Korea':
        return 'Asia'
    if country_name == "Saint Vincent and Grenadines":
        return "North America"
    country_code = pc.country_name_to_country_alpha2(country_name)
    continent_code = pc.country_alpha2_to_continent_code(country_code)
    continent_name = pc.convert_continent_code_to_continent_name(continent_code)
    return continent_name

def encode_bike():
    bike_df = pd.read_csv('input/Bike-Sharing-Dataset/day.csv')
    bike_df['dteday'] = pd.to_datetime(bike_df['dteday'])
    bike_df['day_number'] = bike_df['dteday'].dt.day
    bike_df = bike_df.drop(['instant', 'dteday'], axis=1) # Zach: I removed 'casual' and 'registered' because we should not drop those! We have to do analysis with them...
    return bike_df

def encode_suicide():
    suicide_df = pd.read_csv('input/master.csv')
    continents = ['Asia', 'North America', 'Europe', 'South America', 'Africa', 'Oceania']
    suicide_df['continent'] = suicide_df['country'].apply(country_to_continent)
    y = pd.get_dummies(suicide_df['continent'], prefix="Continent")
    suicide_df = pd.concat([suicide_df, y], axis=1)
    sexkeys = {'male' : 0, 'female' : 1}
    suicide_df['sex'] = suicide_df['sex'].apply(lambda sex: sexkeys[sex])
    agekeys = {'5-14 years' :1,'15-24 years' :2,'25-34 years' :3,'35-54 years' :4,'55-74 years' :5,'75+ years' :6}
    suicide_df['age'] = suicide_df['age'].apply(lambda agerange: agekeys[agerange])
    genkeys = {'G.I. Generation' :1, 'Silent' :2, 'Boomers' :3, 'Generation X' :4, 'Millenials' :5, 'Generation Z' :6 }
    suicide_df['generation'] = suicide_df['generation'].apply(lambda generation: genkeys[generation])
    suicide_df = suicide_df[["Continent_Africa","Continent_Asia", "Continent_Europe", "Continent_North America", "Continent_Oceania", "Continent_South America","year", "sex", "age", "population", " gdp_for_year ($) ", "gdp_per_capita ($)", "generation", "suicides_no", "suicides/100k pop"]]
    return suicide_df

def encode_video():
    # Handling Categorical Features of Online Video Dataset
    video_df = pd.read_csv('input/online_video_dataset/transcoding_mesurment.tsv',sep='\t')
    y = pd.get_dummies(video_df['codec'], prefix='codec')
    video_df = pd.concat([video_df, y], axis=1)
    y = pd.get_dummies(video_df['o_codec'], prefix='o_codec')
    video_df = pd.concat([video_df, y], axis=1)
    video_df = video_df[['duration', 'codec_flv', 'codec_h264',
    'codec_mpeg4', 'codec_vp8', 'height', 'width', 'bitrate', 'framerate', 'i', 'p', 'b',
    'frames', 'i_size', 'p_size', 'b_size', 'size',
    'o_codec_flv', 'o_codec_h264', 'o_codec_mpeg4', 'o_codec_vp8',
    'o_bitrate', 'o_framerate', 'o_width', 'o_height', 'umem', 'utime']]
    return video_df




def q7_helper():
    return encode_bike(), encode_suicide(), encode_video()

def scale_bike():
    return encode_bike()

def scale_suicide():
    suicide_df_final = encode_suicide()
    scaler = StandardScaler()
    suicide_df_final[' gdp_for_year ($) '] = suicide_df_final[' gdp_for_year ($) '].apply(lambda x: float(x.split()[0].replace(',', '')))
    # Standardize feature columns of suicide data.
    suicide_df_final[['year', 'population', ' gdp_for_year ($) ', 'gdp_per_capita ($)']] = scaler.fit_transform(suicide_df_final[['year', 'population', ' gdp_for_year ($) ', 'gdp_per_capita ($)']])
    return suicide_df_final

def scale_video():
    video_df_final = encode_video()
    scaler = StandardScaler()
    video_df_final[['duration', 'height', 'width', 'bitrate','framerate', 'i', 'p', 'b', 'frames', 'i_size', 'p_size', 'b_size','size', 'o_bitrate', 'o_framerate', 'o_width', 'o_height']] = scaler.fit_transform(video_df_final[['duration', 'height', 'width','bitrate', 'framerate', 'i', 'p', 'b', 'frames', 'i_size', 'p_size', 'b_size', 'size','o_bitrate', 'o_framerate', 'o_width', 'o_height']])
    return video_df_final

def q8_helper():
    return scale_bike(), scale_suicide(), scale_video()

def separate_target(df,target,drop_other=[]):
    for other in drop_other:
        df = df.drop([other], axis=1)
    return df.drop(target, axis=1), df[[target]]


# def suicide_df_data_labels():
#     suicide_df_final = scale_suicide()
#     suicide_df_final_features = suicide_df_final.drop(['suicides_no', 'suicides/100k pop'], axis=1)
#     suicide_df_final_target = suicide_df_final[['suicides_no', 'suicides/100k pop']]
#     return suicide_df_final_features, suicide_df_final_target


# def video_df_data_labels():
#     video_df_final = scale_video()
#     video_df_final_features = video_df_final.drop(['umem', 'utime'], axis=1)
#     video_df_final_target = video_df_final[['umem', 'utime']]
#     return video_df_final_features, video_df_final_target


# def bike_df_data_labels():
#     bike_df_final = scale_bike()
#     return separate_target(bike_df_final,'cnt')

def bike_df_cnt_data_labels():
    bike_df_final = scale_bike()
    return separate_target(bike_df_final,'cnt',drop_other=['casual','registered'])

# def bike_df_registered_data_labels():
#     bike_df_final = scale_bike()
#     return separate_target(bike_df_final,'registered',drop_other=['cnt','casual'])

# def bike_df_casual_data_labels():
#     bike_df_final = scale_bike()
#     return separate_target(bike_df_final,'casual',drop_other=['cnt','registered'])



# def suicide_total_df_data_labels():
#     suicide_df_final = scale_suicide()
#     return separate_target(suicide_df_final,'suicides_no',drop_other=['suicides/100k pop'])


def suicide_per_100k_df_data_labels():
    suicide_df_final = scale_suicide()
    return separate_target(suicide_df_final,'suicides/100k pop',drop_other=['suicides_no'])


# def video_df_umem_data_labels():
#     video_df_final = scale_video()
#     return separate_target(video_df_final,'umem',drop_other=['utime'])

def video_df_utime_data_labels():
    video_df_final = scale_video()
    return separate_target(video_df_final,'utime',drop_other=['umem'])

def get_all_data_and_labels():
    return {'bike_cnt' : bike_df_cnt_data_labels(),'video_utime' : video_df_utime_data_labels(),'suicide_per_100k' : suicide_per_100k_df_data_labels()}

def bike_df_cnt_data_labels_no_scale():
    bike_df_final = encode_bike()
    return separate_target(bike_df_final,'cnt',drop_other=['casual','registered'])

def suicide_per_100k_df_data_labels_no_scale():
    suicide_df_final = encode_suicide()
    suicide_df_final[' gdp_for_year ($) '] = suicide_df_final[' gdp_for_year ($) '].apply(lambda x: float(x.split()[0].replace(',', '')))
    return separate_target(suicide_df_final,'suicides/100k pop',drop_other=['suicides_no'])

def video_df_utime_data_labels_no_scale():
    video_df_final = encode_video()
    return separate_target(video_df_final,'utime',drop_other=['umem'])

def get_all_data_and_labels_no_scale():
    return {'bike_cnt' : bike_df_cnt_data_labels_no_scale(),'video_utime' : video_df_utime_data_labels_no_scale(),'suicide_per_100k' : suicide_per_100k_df_data_labels_no_scale()}

def get_all_data_and_labels_selected(pruning_for='linear'):
    data_and_labels = get_all_data_and_labels()
    if pruning_for=='linear':
        drop = { 'bike_cnt' : ['day_number'], 'video_utime': ['codec_flv', 'duration', 'b_size', 'b' ], 'suicide_per_100k': ['population', 'gdp_per_capita ($)']} # determined based on LASSO
    elif pruning_for=='nn':
        drop = { 'bike_cnt' : [], 'video_utime': [ ], 'suicide_per_100k': ['year', 'Continent_Oceania']}
    elif pruning_for=='forest':
        drop = { 'bike_cnt' : [], 'video_utime': [ ], 'suicide_per_100k': []}
    else:
        drop = {}
    for k in drop:
        features,target = data_and_labels[k]
        features.drop(drop[k], axis=1,inplace=True)
    return data_and_labels

def get_all_data_and_labels_selected_no_scale(pruning_for='linear'):
    data_and_labels = get_all_data_and_labels_no_scale()
    if pruning_for=='linear':
        drop = { 'bike_cnt' : ['day_number'], 'video_utime': ['codec_flv', 'duration', 'b_size', 'b' ], 'suicide_per_100k': ['population', 'gdp_per_capita ($)']} # determined based on LASSO
    elif pruning_for=='nn':
        drop = { 'bike_cnt' : [], 'video_utime': [ ], 'suicide_per_100k': ['year', 'Continent_Oceania']}
    elif pruning_for=='forest':
        drop = { 'bike_cnt' : [], 'video_utime': [ ], 'suicide_per_100k': []}
    else:
        drop = {}
    for k in drop:
        features,target = data_and_labels[k]
        features.drop(drop[k], axis=1,inplace=True)
    return data_and_labels

def q12_helper():
    bike_df_final, suicide_df_final, video_df_final = q7_helper()
    suicide_df_final[' gdp_for_year ($) '] = suicide_df_final[' gdp_for_year ($) '].apply(lambda x: float(x.split()[0].replace(',', '')))
    bike_df_final_features = bike_df_final.drop('cnt', axis=1)
    bike_df_final_target = bike_df_final[['cnt']]
    suicide_df_final_features = suicide_df_final.drop(['suicides_no', 'suicides/100k pop'], axis=1)
    suicide_df_final_target = suicide_df_final[['suicides_no', 'suicides/100k pop']]
    video_df_final_features = video_df_final.drop(['umem', 'utime'], axis=1)
    video_df_final_target = video_df_final[['umem', 'utime']]

    return bike_df_final_features, bike_df_final_target, suicide_df_final_features, suicide_df_final_target, video_df_final_features, video_df_final_target
