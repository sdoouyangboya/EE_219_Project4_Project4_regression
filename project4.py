from helpers import load_datasets, q4_helper, q5_helper, q6_helper, q7_helper, q8_helper, q12_helper, get_all_data_and_labels, get_all_data_and_labels_selected, get_all_data_and_labels_selected_no_scale
from pandas_profiling import ProfileReport
import pandas as pd
import pycountry_convert as pc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from logger import logged
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.tree import export_graphviz
from caching import cached
from skopt import BayesSearchCV
from catboost import Pool, CatBoostRegressor
from skopt.space import Real, Categorical, Integer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import statsmodels.api as sm
import warnings
import graphviz


import warnings
warnings.simplefilter("ignore")
# from sklearn.exceptions import ConvergenceWarning
# warnings.filterwarnings("ignore", category=ConvergenceWarning)


#TODO: UPLOAD TO OVERLEAF
def q1_q2():
    bike_df, suicide_df, video_df = load_datasets()
    profile_bike = ProfileReport(bike_df, title="Bike Sharing Report")
    profile_bike.to_file("output/Bike_Sharing_Report.html")
    profile_suicide = ProfileReport(suicide_df, title="Suicide Rates Overview Report")
    profile_suicide.to_file("output/Suicide Rates Overview Report.html")
    profile_video = ProfileReport(video_df, title="Video Transcoding Time Report")
    profile_video.to_file("output/Video Transcoding Time Report.html")

#TODO: CODE AND OVERLEAF
def q3():
    bike_df, suicide_df, video_df = load_datasets()
    fig, ax = plt.subplots()
    suicide_plot = sns.boxplot(data=suicide_df,x="suicides/100k pop",y="generation",hue="age",fliersize=0.001,ax=ax)
    ax.set_title(f"Suicides/100k pop vs. Generation and Age Range")
    fig.tight_layout(pad=0.25)
    plt.savefig(f"figures/q3_suicides.pdf")
    # plt.show()
    bike_df['weathersit'] = bike_df['weathersit'].astype('category')
    fig, axs = plt.subplots(2,3,figsize=(5* len(set(bike_df.weathersit.values)), 10))
    for holiday in (0,1):
        for j,weathersit in enumerate(set(bike_df.weathersit.values)):
            ax = axs[holiday,j]
            filtered = bike_df[(bike_df.weathersit == weathersit) & (bike_df.holiday == holiday)]
            if len(filtered) > 0:
                sns.boxplot(data=filtered,y="cnt",x="season",hue="weekday",ax=ax)
            ax.set_title(f"Bike Rentals vs. Season, Weekday\n{'Holiday' if holiday else 'Non-Holiday'}, Weather={weathersit}")
    fig.tight_layout(pad=0.25)
    plt.savefig(f"figures/q3_bikes.pdf")
    # plt.show()
    # "codec", "bitrate", "framerate", "o_codec", "o_bitrate",
    framerates = sorted([ float(framerate) for framerate in set(video_df['o_framerate'].values)])
    bitrates = sorted([float(bitrate) for bitrate in set(video_df['o_bitrate'].values)])
    for target in ('utime','umem'):
        fig, axs = plt.subplots(len(framerates),len(bitrates),figsize=(5*len(bitrates),5*len(framerates)))
        for (i,framerate) in enumerate(framerates):
            for (j, bitrate) in enumerate(bitrates):
                ax = axs[i,j]
                filtered = video_df[(video_df['o_framerate'] == framerate) & (video_df['o_bitrate'] == bitrate)]
                if len(filtered) > 0:
                    sns.boxplot(data=filtered,x=target,y="codec",hue="o_codec",fliersize=0.001,ax=ax)
                    # sns.boxplot(data=filtered,x=target,y="codec",hue="o_codec",fliersize=0.001)
                ax.set_title(f"Encoding {target} vs. Codec\nFramerate={framerate}, Bitrate={bitrate}")
        fig.tight_layout(pad=0.25)
        plt.savefig(f"figures/q3_videos_{target}.pdf")



#TODO: UPLOAD TO OVERLEAF
def q4():
    max_year, smallest_month, max_month, count_list = q4_helper()
    for (i,j),current_list in count_list.items():
        plt.plot(list(range(len(current_list))), current_list)
        plt.xlabel('Day')
        plt.ylabel('Count')
        if i == 0:
            year = 2011
        else:
            year = 2012
        plt.title(f'Year: {year}, Month: {j} Count Number per Day')
        plt.savefig(f"figures/q4_{year}_{j}_count_number_per_day.pdf")
        plt.clf()

#TODO: UPLOAD TO OVERLEAF
def q5():
    suicide_df, ten_largest = q5_helper()
    for country_name in ten_largest:
        suicide_df_country = suicide_df[suicide_df['country'] == country_name]
        sns_plot = sns.relplot(data=suicide_df_country, x="year", y ="suicides/100k pop" , hue="age", col="sex")
        sns_plot.savefig(f"figures/q5_{country_name}_suicides100kpop_vs_year")


def q5_one_plot():
    suicide_df, ten_largest = q5_helper()
    suicide_df_country = suicide_df[suicide_df['country'].isin(ten_largest)]
    sns_plot = sns.relplot(data=suicide_df_country, x="year", y ="suicides/100k pop" , hue="age", col="sex",row='country',row_order=ten_largest)
    sns_plot.savefig(f"figures/q5__all_suicides100kpop_vs_year")

#TODO: UPLOAD TO OVERLEAF
@logged
def q6():
    video_df_utime = q6_helper()
    mean_utime  = video_df_utime['utime'].mean()
    median_utime =  video_df_utime['utime'].median()
    hist = video_df_utime.plot.hist(bins=50, title=f"Distribution of Video Transcoding Times\nMean={mean_utime : 0.3f}, Median={median_utime : 0.3f}", legend=False, figsize=(7.5, 7.5))
    hist.set_xlabel("Video Transcoding Times")
    fig = hist.get_figure()
    fig.savefig("figures/q6_distribution_of_video_transcoding_times")
    mean, median = video_df_utime['utime'].mean(), video_df_utime['utime'].median()
    print(f"Mean Transcoding Time: {mean}, Median Transcoding Time: {median}")

#TODO: UPLOAD TO OVERLEAF
def q7():
    bike_df, suicide_df, video_df = q7_helper()
    return bike_df, suicide_df, video_df

#TODO: UPLOAD TO OVERLEAF
def q8():
    bike_df_final, suicide_df_final, video_df_final = q8_helper()
    return bike_df_final, suicide_df_final, video_df_final

#TODO: CODE AND OVERLEAF
@logged
def q9():
    # Need to analyze relevant features and decide which ones to return.
    data_and_labels = get_all_data_and_labels() #{'bike_cnt' : bike_df_cnt_data_labels(), 'bike_registered' : bike_df_registered_data_labels(), 'bike_casual' : bike_df_casual_data_labels(),  'video_umem' : video_df_umem_data_labels(),'video_utime' : video_df_utime_data_labels(),'suicide_per_100k' : suicide_per_100k_df_data_labels(),'suicide_total' : suicide_total_df_data_labels()}
    pruned_data = {}
    model = LinearRegression()
    kf = KFold(n_splits=10)
    for k, (features, target) in data_and_labels.items():
        all_features = features.columns
        f_scores, pvals =  f_regression(features.values, target.values.ravel())
        pruned_features = [feature for (feature,pval) in zip(features.columns,pvals) if pval > 0.05]

        test_score_before = np.mean(-cross_validate(model, features, target, scoring='neg_root_mean_squared_error', return_train_score=True, cv=kf)['test_score'])

        features.drop(pruned_features, axis=1,inplace=True)
        pruned_data[k] = (features, target)
        test_score_after = np.mean(-cross_validate(model, features, target, scoring='neg_root_mean_squared_error', return_train_score=True, cv=kf)['test_score'])

        print(f"{k}: After removing {len(pruned_features)}/{len(all_features)} insignificant features, Linear Regression RMSE on test set changes: {test_score_before} -> {test_score_after}")
    return pruned_data
    # return data_and_labels
        # features = features[[ feature for (feature,pval) in zip(features.columns,pvals) if pval < 0.05 ]]
        # data_and_labels[k] = (features,target)
     # k = 'video_umem'
    # features,target = data_and_labels[k]
    # mutual_infos = {}
    # f_regressions = {}
    # mutual_infos[k]= mutual_info_regression(features.values, target.values.ravel())
    # f_regressions[k] = f_regression(features.values, target.values.ravel())
    # Need to read through the data to determine the relevant features. Return the modified data.

    # return bike_df_final_features, bike_df_final_target, suicide_df_final_features, suicide_df_final_target, video_df_final_features, video_df_final_target

@logged
def q10():
# drop the feature with lowest mutual information iteratively as long as out-of-sample performance improves
    def stepwise_backward_selection_mutual_info(features,target,model,linear=False,n_splits=10):
        kf = KFold(n_splits=n_splits)
        test_score = np.mean(cross_validate(model, features.values, target.values.ravel(), scoring='neg_root_mean_squared_error', return_train_score=True, cv=kf)['test_score'])
        while True:
            if linear:
                weakest_feature = weakest_feature_linear(features,target)
            else:
                weakest_feature = weakest_feature_nonlinear(features,target)
            if weakest_feature is None:
                break
            new_features = features.drop([weakest_feature],axis=1)
            new_score  = np.mean(cross_validate(model, new_features.values, target.values.ravel(), scoring='neg_root_mean_squared_error', return_train_score=True, cv=kf)['test_score'])
            if new_score < test_score:
                break
            test_score = new_score
            features = new_features
        return features

    def weakest_feature_linear(features,target):
        f_scores, pvals =  f_regression(features.values, target.values.ravel())
        if max(pvals) < 0.05:
            return None
        return features.columns[np.argmax(pvals)]

    def weakest_feature_nonlinear(features,target):
        mutual_info = mutual_info_regression(features.values, target.values.ravel())
        return features.columns[np.argmin(mutual_info)]

    # @cached
    def prune_all(data_and_labels,model,linear=False):
        for k, (features, target) in data_and_labels.items():
            pruned_features = stepwise_backward_selection_mutual_info(features,target,model,linear=linear)
            data_and_labels[k] = (pruned_features,target)
        return data_and_labels


    # data_and_labels = get_all_data_and_labels()
    # k='video_umem'
    # features,target = data_and_labels[k]

    models = (LinearRegression(), Lasso(alpha = 0.5),Ridge(alpha = 0.5),
              MLPRegressor(), RandomForestRegressor(n_estimators = 10, max_depth = 30, max_features= "auto"))
    for model in models:
        linear = 'linear_model' in f"{type(model)}"
        # data_and_labels = model_selection_all(model=model,linear=linear)
        data_and_labels = get_all_data_and_labels()
        print(f"Model is '{model}', selecting using {'p-Values of F-statistics' if linear else 'Mutual Information'}")
        print(f"Number of features BEFORE model selection:\n{ {k : len(features.columns) for k, (features, target) in data_and_labels.items()} }")
        og_cols = {k : features.columns for k, (features,target) in data_and_labels.items()}
        data_and_labels = prune_all(data_and_labels,model,linear=linear)
        removed_cols = {k : "'{}'".format("', '".join(set(og_cols[k]) - set(features.columns))) for  k, (features,target) in data_and_labels.items()}
        for k, cols in removed_cols.items():
            print(f"\tFrom {k} pruned: {cols}")
        print(f"Number of features AFTER model selection:\n{ {k : len(features.columns) for k, (features, target) in data_and_labels.items()} }\n")

#TODO: UPLOAD TO OVERLEAF, RE-RUN WITH PRUNED DATA
@logged
def q10_q11_13_linear():
    data_and_labels = get_all_data_and_labels_selected(pruning_for='linear')
    # data_and_labels = get_all_data_and_labels()
    model = LinearRegression()
    kf = KFold(n_splits=10)
    print("Linear Regression Results: ")
    for k, (features, target) in data_and_labels.items():
        scores = cross_validate(model, features.values, target.values.ravel(), scoring='neg_root_mean_squared_error', return_train_score=True, cv=kf)
        avg_train_score = np.mean(-scores['train_score'])
        avg_test_score = np.mean(-scores['test_score'])
        print(f"{k}: Avg. Training RMSE Error: {avg_train_score}, Avg. Test RMSE Error: {avg_test_score}")

@logged
def q10_q11_13_linear_shuffle():
    data_and_labels = get_all_data_and_labels_selected(pruning_for='linear')
    # data_and_labels = get_all_data_and_labels()
    model = LinearRegression()
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    print("Linear Regression Shuffling Data Results: ")
    for k, (features, target) in data_and_labels.items():
        scores = cross_validate(model, features.values, target.values.ravel(), scoring='neg_root_mean_squared_error', return_train_score=True, cv=kf)
        avg_train_score = np.mean(-scores['train_score'])
        avg_test_score = np.mean(-scores['test_score'])
        print(f"{k}: Avg. Training RMSE Error: {avg_train_score}, Avg. Test RMSE Error: {avg_test_score}")

#TODO: UPLOAD TO OVERLEAF, RE-RUN WITH PRUNED DATA
@logged
def q10_q11_13_lasso():
    data_and_labels = get_all_data_and_labels_selected(pruning_for='linear')
    alpha = [0.1, 0.2, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0]
    best_alpha = {}
    kf = KFold(n_splits=10)
    for parameter in alpha:
        model = Lasso(alpha = parameter)
        print(f"Lasso Results for {parameter}: ")
        for k, (features, target) in data_and_labels.items():
            scores = cross_validate(model, features.values, target.values.ravel(), scoring='neg_root_mean_squared_error', return_train_score=True, cv=kf)
            avg_train_score = np.mean(-scores['train_score'])
            avg_test_score = np.mean(-scores['test_score'])
            # The smaller the RMSE, the better.
            if k not in best_alpha: best_alpha[k] = (parameter, avg_train_score, avg_test_score)
            if best_alpha[k][2] > avg_test_score:
                best_alpha[k] = (parameter, avg_train_score, avg_test_score)
            print(f"{k}: Avg. Training RMSE Error: {avg_train_score}, Avg. Test RMSE Error: {avg_test_score}")
    print("Best Alphas for each Dataset: ")
    for key, value in best_alpha.items():
        print(key, value)
    pass 

@logged
def q10_q11_13_lasso_shuffle():
    data_and_labels = get_all_data_and_labels_selected(pruning_for='linear')
    alpha = [0.1, 0.2, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0]
    best_alpha = {}
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    for parameter in alpha:
        model = Lasso(alpha = parameter)
        print(f"Lasso Shuffling Data Results for {parameter}: ")
        for k, (features, target) in data_and_labels.items():
            scores = cross_validate(model, features.values, target.values.ravel(), scoring='neg_root_mean_squared_error', return_train_score=True, cv=kf)
            avg_train_score = np.mean(-scores['train_score'])
            avg_test_score = np.mean(-scores['test_score'])
            # The smaller the RMSE, the better.
            if k not in best_alpha: best_alpha[k] = (parameter, avg_train_score, avg_test_score)
            if best_alpha[k][2] > avg_test_score:
                best_alpha[k] = (parameter, avg_train_score, avg_test_score)
            print(f"{k}: Avg. Training RMSE Error: {avg_train_score}, Avg. Test RMSE Error: {avg_test_score}")
    print("Best Alphas for each Dataset: ")
    for key, value in best_alpha.items():
        print(key, value)
    pass 

#TODO: UPLOAD TO OVERLEAF, RE-RUN WITH PRUNED DATA
@logged
def q10_q11_13_ridge():
    data_and_labels = get_all_data_and_labels_selected(pruning_for='linear')
    alpha = [0.1, 0.2, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0]
    best_alpha = {}
    kf = KFold(n_splits=10)
    for parameter in alpha:
        model = Ridge(alpha = parameter)
        print(f"Ridge Results for {parameter}: ")
        for k, (features, target) in data_and_labels.items():
            scores = cross_validate(model, features.values, target.values.ravel(), scoring='neg_root_mean_squared_error', return_train_score=True, cv=kf)
            avg_train_score = np.mean(-scores['train_score'])
            avg_test_score = np.mean(-scores['test_score'])
            # The smaller the RMSE, the better.
            if k not in best_alpha: best_alpha[k] = (parameter, avg_train_score, avg_test_score)
            if best_alpha[k][2] > avg_test_score:
                best_alpha[k] = (parameter, avg_train_score, avg_test_score)
            print(f"{k}: Avg. Training RMSE Error: {avg_train_score}, Avg. Test RMSE Error: {avg_test_score}")
    print("Best Alphas for each Dataset: ")
    for key, value in best_alpha.items():
        print(key, value)
    pass 

@logged
def q10_q11_13_ridge_shuffle():
    data_and_labels = get_all_data_and_labels_selected(pruning_for='linear')
    alpha = [0.1, 0.2, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0]
    best_alpha = {}
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    for parameter in alpha:
        model = Ridge(alpha = parameter)
        print(f"Ridge Shuffled Data Results for {parameter}: ")
        for k, (features, target) in data_and_labels.items():
            scores = cross_validate(model, features.values, target.values.ravel(), scoring='neg_root_mean_squared_error', return_train_score=True, cv=kf)
            avg_train_score = np.mean(-scores['train_score'])
            avg_test_score = np.mean(-scores['test_score'])
            # The smaller the RMSE, the better.
            if k not in best_alpha: best_alpha[k] = (parameter, avg_train_score, avg_test_score)
            if best_alpha[k][2] > avg_test_score:
                best_alpha[k] = (parameter, avg_train_score, avg_test_score)
            print(f"{k}: Avg. Training RMSE Error: {avg_train_score}, Avg. Test RMSE Error: {avg_test_score}")
    print("Best Alphas for each Dataset: ")
    for key, value in best_alpha.items():
        print(key, value)
    pass 

@logged
def q12_linear():
    # Don't do feature scaling. Load dataset. (Should do feature selection here.)
    data_and_labels = get_all_data_and_labels_selected_no_scale(pruning_for='linear')
    bike_features, bike_targets = data_and_labels['bike_cnt']
    video_features, video_targets = data_and_labels['video_utime']
    suicide_features, suicide_targets = data_and_labels['suicide_per_100k']

    model_bike = LinearRegression()
    model_video = LinearRegression()
    model_suicide = LinearRegression()
    
    # kf = KFold(n_splits=10, shuffle=True, random_state=0)
    kf = KFold(n_splits=10)
    
    scores_bike = cross_validate(model_bike, bike_features.values, bike_targets.values.ravel(), scoring='neg_root_mean_squared_error', return_train_score=True, cv=kf)
    scores_video = cross_validate(model_video, video_features.values, video_targets.values.ravel(), scoring='neg_root_mean_squared_error', return_train_score=True, cv=kf)
    scores_suicides = cross_validate(model_suicide, suicide_features.values, suicide_targets.values.ravel(), scoring='neg_root_mean_squared_error', return_train_score=True, cv=kf)
    
    bike_avg_train_score = np.mean(-scores_bike['train_score'])
    bike_avg_test_score = np.mean(-scores_bike['test_score'])

    video_avg_train_score = np.mean(-scores_video['train_score'])
    video_avg_test_score = np.mean(-scores_video['test_score'])

    suicide_avg_train_score = np.mean(-scores_suicides['train_score'])
    suicide_avg_test_score = np.mean(-scores_suicides['test_score'])  

    print("No Feature Scaling Test for Linear Regression; Using the best models from Feature Scaling")
    print(f"Bike Sharing Dataset: Avg. Training RMSE Error: {bike_avg_train_score}, Avg. Test RMSE Error: {bike_avg_test_score}")
    print(f"Video Dataset: Avg. Training RMSE Error: {video_avg_train_score}, Avg. Test RMSE Error: {video_avg_test_score}")
    print(f"Suicide Dataset: Avg. Training RMSE Error: {suicide_avg_train_score}, Avg. Test RMSE Error: {suicide_avg_test_score}")

@logged
def q12_lasso():
    # Don't do feature scaling. Load dataset. (Should do feature selection here.)
    data_and_labels = get_all_data_and_labels_selected_no_scale(pruning_for='linear')
    bike_features, bike_targets = data_and_labels['bike_cnt']
    video_features, video_targets = data_and_labels['video_utime']
    suicide_features, suicide_targets = data_and_labels['suicide_per_100k']

    model_bike = Lasso(alpha = 2.0)
    model_video = Lasso(alpha = 0.1)
    model_suicide = Lasso(alpha = 0.1)
    
    # kf = KFold(n_splits=10, shuffle=True, random_state=0)
    kf = KFold(n_splits=10)
    
    scores_bike = cross_validate(model_bike, bike_features.values, bike_targets.values.ravel(), scoring='neg_root_mean_squared_error', return_train_score=True, cv=kf)
    scores_video = cross_validate(model_video, video_features.values, video_targets.values.ravel(), scoring='neg_root_mean_squared_error', return_train_score=True, cv=kf)
    scores_suicides = cross_validate(model_suicide, suicide_features.values, suicide_targets.values.ravel(), scoring='neg_root_mean_squared_error', return_train_score=True, cv=kf)
    
    bike_avg_train_score = np.mean(-scores_bike['train_score'])
    bike_avg_test_score = np.mean(-scores_bike['test_score'])

    video_avg_train_score = np.mean(-scores_video['train_score'])
    video_avg_test_score = np.mean(-scores_video['test_score'])

    suicide_avg_train_score = np.mean(-scores_suicides['train_score'])
    suicide_avg_test_score = np.mean(-scores_suicides['test_score'])  

    print("No Feature Scaling Test for Lasso Regression; Using the best models from Feature Scaling")
    print(f"Bike Sharing Dataset: Avg. Training RMSE Error: {bike_avg_train_score}, Avg. Test RMSE Error: {bike_avg_test_score}")
    print(f"Video Dataset: Avg. Training RMSE Error: {video_avg_train_score}, Avg. Test RMSE Error: {video_avg_test_score}")
    print(f"Suicide Dataset: Avg. Training RMSE Error: {suicide_avg_train_score}, Avg. Test RMSE Error: {suicide_avg_test_score}")


@logged
def q12_ridge():
    # Don't do feature scaling. Load dataset. (Should do feature selection here.)
    data_and_labels = get_all_data_and_labels_selected_no_scale(pruning_for='linear')
    bike_features, bike_targets = data_and_labels['bike_cnt']
    video_features, video_targets = data_and_labels['video_utime']
    suicide_features, suicide_targets = data_and_labels['suicide_per_100k']

    model_bike = Ridge(alpha = 0.5)
    model_video = Ridge(alpha = 5.0)
    model_suicide = Ridge(alpha = 10.0)
    
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    # kf = KFold(n_splits=10)
    
    scores_bike = cross_validate(model_bike, bike_features.values, bike_targets.values.ravel(), scoring='neg_root_mean_squared_error', return_train_score=True, cv=kf)
    scores_video = cross_validate(model_video, video_features.values, video_targets.values.ravel(), scoring='neg_root_mean_squared_error', return_train_score=True, cv=kf)
    scores_suicides = cross_validate(model_suicide, suicide_features.values, suicide_targets.values.ravel(), scoring='neg_root_mean_squared_error', return_train_score=True, cv=kf)
    
    bike_avg_train_score = np.mean(-scores_bike['train_score'])
    bike_avg_test_score = np.mean(-scores_bike['test_score'])

    video_avg_train_score = np.mean(-scores_video['train_score'])
    video_avg_test_score = np.mean(-scores_video['test_score'])

    suicide_avg_train_score = np.mean(-scores_suicides['train_score'])
    suicide_avg_test_score = np.mean(-scores_suicides['test_score'])  

    print("No Feature Scaling Test for Ridge Regression; Using the best models from Feature Scaling")
    print(f"Bike Sharing Dataset: Avg. Training RMSE Error: {bike_avg_train_score}, Avg. Test RMSE Error: {bike_avg_test_score}")
    print(f"Video Dataset: Avg. Training RMSE Error: {video_avg_train_score}, Avg. Test RMSE Error: {video_avg_test_score}")
    print(f"Suicide Dataset: Avg. Training RMSE Error: {suicide_avg_train_score}, Avg. Test RMSE Error: {suicide_avg_test_score}")

@logged
def q14():
    degrees = [2]
    kf = KFold(n_splits=10)
    for degree in degrees:
        data_and_labels = get_all_data_and_labels_selected(pruning_for='linear')
        polynomial_features = PolynomialFeatures(degree=degree)  
        for k, (features, target) in data_and_labels.items():
            poly_features = polynomial_features.fit_transform(features.values)
            mod = sm.OLS(target.values, poly_features)
            results = mod.fit()
            print(results.summary()) 

@logged
def q14_q15_16_poly():
    alpha = 1000.0
    degrees = [2]
    kf = KFold(n_splits=10)
    model = Ridge(alpha = alpha)
    for degree in degrees:
        data_and_labels = get_all_data_and_labels_selected(pruning_for='linear')
        print("Polyniomial Linear Regression Results For {} Degree:".format(degree))
        polynomial_features = PolynomialFeatures(degree=degree)  
        for k, (features, target) in data_and_labels.items():
            if k == "video_utime":

                
                
                poly_features = polynomial_features.fit_transform(features.values)
                print(np.shape(poly_features))
                scores = cross_validate(model, poly_features, target.values.ravel(), 
                                        scoring='neg_root_mean_squared_error', return_train_score=True, cv=kf)
                avg_train_score = np.mean(-scores['train_score'])
                avg_test_score = np.mean(-scores['test_score'])
                print(f"{k}: Avg. Training RMSE Error: {avg_train_score}, Avg. Test RMSE Error: {avg_test_score}")
            
        data_and_labels = get_all_data_and_labels_selected(pruning_for='linear')
        print("Polyniomial Linear Regression Results For {} Degree:".format(degree))
        polynomial_features = PolynomialFeatures(degree=degree)  
        for k, (features, target) in data_and_labels.items():
            if k == "video_utime":
                #out = np.zeros_like(features.values)
                #mask = np.zeros_like(features.values)
                #for i in range(np.size(out, 1)):
                #    if i == (3 or 4 or 5 or 6 or 7 or 8 or 9 or 10 or 11 or 12 or 18 or 19 or 20 or 21):
                #            mask[i] = np.ones_like(np.size(out, 0))
                            
                # np.reciprocal(features.values, where=np.multiply(features.values,mask) > 0.0, out=out)

                # features = np.hstack((features.values, out))
                
                temp = features.values
                temp1 = temp[:,12]
                temp2 = temp[:,6]
                temp3 = temp[:,9]
                temp4 = np.multiply(np.multiply(temp1,temp2), np.reciprocal(temp3, where=temp3 > 0.0))
                
                poly_features = polynomial_features.fit_transform(features.values)
                poly_features = np.column_stack((poly_features, temp4))
                # poly_features = np.concatenate((poly_features, temp4), axis=1)
                # poly_features = np.hstack((poly_features, temp4))
                print(np.shape(poly_features))
                scores = cross_validate(model, poly_features, target.values.ravel(), 
                                        scoring='neg_root_mean_squared_error', return_train_score=True, cv=kf)
                avg_train_score = np.mean(-scores['train_score'])
                avg_test_score = np.mean(-scores['test_score'])
                print(f"{k}: Avg. Training RMSE Error: {avg_train_score}, Avg. Test RMSE Error: {avg_test_score}")

                #['codec_h264',
                #'codec_mpeg4', 'codec_vp8', 'height', 'width', 'bitrate', 'framerate', 'i', 'p',
                #'frames', 'i_size', 'p_size','size',
                #'o_codec_flv', 'o_codec_h264', 'o_codec_mpeg4', 'o_codec_vp8',
                #'o_bitrate', 'o_framerate', 'o_width', 'o_height','utime']
@logged
def q17_18_19_20_NN():
    kf = KFold(n_splits=10)
    model = MLPClassifier(solver='adam',
                          activation = 'relu',
                          alpha=0.0001, 
                          hidden_layer_sizes=(100, 50),
                          learning_rate = 'adaptive',
                          learning_rate_init = 0.001,
                          batch_size='auto',
                          verbose = False,
                          max_iter = 200,
                          random_state=1)
    data_and_labels = get_all_data_and_labels_selected(pruning_for='nn')
    print("Fully Connected Network Results:")
    for k, (features, target) in data_and_labels.items():
        print(f"Cross-validating on {k}")
        scores = cross_validate(model, features.values, target.values.ravel(), 
                                scoring='neg_root_mean_squared_error', return_train_score=True, cv=kf)
        avg_train_score = np.mean(-scores['train_score'])
        avg_test_score = np.mean(-scores['test_score'])
        print(f"{k}: Avg. Training RMSE Error: {avg_train_score}, Avg. Test RMSE Error: {avg_test_score}")


@logged
def q18():
    data_and_labels = get_all_data_and_labels_selected(pruning_for='nn')
    features, target = data_and_labels['bike_cnt'] #use the bike dataset to determine model parameters.
    kf = KFold(n_splits=10)
    alphas = [10**(-k) for k in range(1,5)]
    depths = list(range(2,20,5))
    widths = [20,50,100]
    networksizes = [ tuple(width for _ in range(depth) ) for depth in depths for width in widths] #e.g. [20, 20], [50,50,50,...,50], [100,100,100,100,100]
    test_performance = {}
    for alpha in alphas:
        for networksize in networksizes:
            model = MLPRegressor(solver='adam',
                                  activation = 'relu',
                                  alpha=alpha,
                                  hidden_layer_sizes=networksize,
                                  learning_rate = 'adaptive',
                                  learning_rate_init = 0.001,
                                  batch_size='auto',
                                  verbose = False,
                                  max_iter = 200,
                                  random_state=1)
            scores = cross_validate(model, features.values, target.values.ravel(),scoring='neg_root_mean_squared_error', return_train_score=True, cv=kf)
            # avg_train_score = np.mean(-scores['train_score'])
            avg_test_score = np.mean(-scores['test_score'])
            test_performance[alpha,networksize] = avg_test_score
            print(f"{alpha=}, {networksize=}, {avg_test_score=}")
    best_alpha, best_networksize = min(test_performance,key=test_performance.get)
    print(f"Best Average RMSE: {test_performance[best_alpha, best_networksize] : .02f}\nBest network parameters are:\nalpha={best_alpha}\nLayers=[{', '.join(str(layer) for layer in best_networksize)}]")

#TODO: UPLOAD TO OVERLEAF, RE-RUN WITH PRUNED DATA
@logged
def q21_22():
    data_and_labels = get_all_data_and_labels_selected(pruning_for='forest')
    kf = KFold(n_splits=10)
    max_features = [0.2, 0.4, 0.6, 0.8, "auto", "sqrt", "log2"]
    number_trees = [10, 50, 100, 150, 200]
    max_depth = [1, 2, 3, 4, 5, 7]
    best_parameters = {}
    for i in max_features:
        for j in number_trees:
            for k in max_depth:
                model = RandomForestRegressor(n_estimators = j, max_depth = k, max_features= i)
                print(f"Random Forest for Max Number Features: {i}, Number of Trees: {j}, Depth of each Tree: {k}")
                for data_set_name, (features, target) in data_and_labels.items():
                    scores = cross_validate(model, features.values, target.values.ravel(), scoring='neg_root_mean_squared_error', return_train_score=True, cv=kf)
                    avg_train_score = np.mean(-scores['train_score'])
                    avg_test_score = np.mean(-scores['test_score'])
                    # The smaller the RMSE, the better.
                    if data_set_name not in best_parameters: best_parameters[data_set_name] = (i, j, k, avg_train_score, avg_test_score)
                    if best_parameters[data_set_name][4] > avg_test_score:
                        best_parameters[data_set_name] = (i, j, k, avg_train_score, avg_test_score)
                    print(f"{data_set_name}: Avg. Training RMSE Error: {avg_train_score}, Avg. Test RMSE Error: {avg_test_score}")
                pass
    print("Best Max Number Features, Number of Trees, Depth of Each Tree for each Dataset")
    for key, value in best_parameters.items():
        print(key, value)
    pass

@logged
def q21_22_oob():
    data_and_labels = get_all_data_and_labels_selected(pruning_for='forest')
    bike_features, bike_targets = data_and_labels['bike_cnt']
    video_features, video_targets = data_and_labels['video_utime']
    suicide_features, suicide_targets = data_and_labels['suicide_per_100k']

    bike_model = RandomForestRegressor(n_estimators = 100, max_depth = 7, max_features= 0.8, oob_score=True)
    bike_model.fit(bike_features.values, bike_targets.values.ravel())

    suicide_model = RandomForestRegressor(n_estimators = 200, max_depth = 7, max_features= 0.8, oob_score=True)
    suicide_model.fit(suicide_features.values, suicide_targets.values.ravel())
    video_model = RandomForestRegressor(n_estimators = 50, max_depth = 7, max_features= 0.4, oob_score=True)
    video_model.fit(video_features.values, video_targets.values.ravel())


    print('OOB Error for each Dataset')
    print(f"Bike Dataset: {1 - bike_model.oob_score_}")
    print(f"Suicide Dataset: {1 - suicide_model.oob_score_}")
    print(f"Video Dataset: {1 - video_model.oob_score_}")

@logged
def q21_22_r():
    data_and_labels = get_all_data_and_labels_selected(pruning_for='forest')
    bike_features, bike_targets = data_and_labels['bike_cnt']
    video_features, video_targets = data_and_labels['video_utime']
    suicide_features, suicide_targets = data_and_labels['suicide_per_100k']

    bike_model = RandomForestRegressor(n_estimators = 100, max_depth = 7, max_features= 0.8, oob_score=True)
    bike_model.fit(bike_features.values, bike_targets.values.ravel())

    suicide_model = RandomForestRegressor(n_estimators = 200, max_depth = 7, max_features= 0.8, oob_score=True)
    suicide_model.fit(suicide_features.values, suicide_targets.values.ravel())
    video_model = RandomForestRegressor(n_estimators = 50, max_depth = 7, max_features= 0.4, oob_score=True)
    video_model.fit(video_features.values, video_targets.values.ravel())

    print('R^2 Error for each Dataset')
    print(f"Bike Dataset: {bike_model.score(bike_features.values, bike_targets.values.ravel())}")
    print(f"Suicide Dataset: {suicide_model.score(suicide_features.values, suicide_targets.values.ravel())}")
    print(f"Video Dataset: {video_model.score(video_features.values, video_targets.values.ravel())}")

    pass

@logged
def q23():
    data_and_labels = get_all_data_and_labels_selected(pruning_for='forest')
    kf = KFold(n_splits=10)
    max_features = [0.2, 0.4, 0.6, 0.8, "auto", "sqrt", "log2"]
    number_trees = [10, 50, 100, 150, 200]
    best_parameters = {}
    for i in max_features:
        for j in number_trees:
            model = RandomForestRegressor(n_estimators = j, max_depth = 4, max_features= i)
            print(f"Random Forest for Max Number Features: {i}, Number of Trees: {j}, Depth of each Tree: 4")
            for data_set_name, (features, target) in data_and_labels.items():
                scores = cross_validate(model, features.values, target.values.ravel(), scoring='neg_root_mean_squared_error', return_train_score=True, cv=kf)
                avg_train_score = np.mean(-scores['train_score'])
                avg_test_score = np.mean(-scores['test_score'])
                # The smaller the RMSE, the better.
                if data_set_name not in best_parameters: best_parameters[data_set_name] = (i, j, avg_train_score, avg_test_score)
                if best_parameters[data_set_name][3] > avg_test_score:
                    best_parameters[data_set_name] = (i, j, avg_train_score, avg_test_score)
                print(f"{data_set_name}: Avg. Training RMSE Error: {avg_train_score}, Avg. Test RMSE Error: {avg_test_score}")
            pass
    print("Best Max Number Features, Number of Trees, Depth of Each Tree for each Dataset")
    for key, value in best_parameters.items():
        print(key, value)
    pass

@logged
def q23_plot():
    data_and_labels = get_all_data_and_labels_selected(pruning_for='forest')
    bike_features, bike_targets = data_and_labels['bike_cnt']
    video_features, video_targets = data_and_labels['video_utime']
    suicide_features, suicide_targets = data_and_labels['suicide_per_100k']
    bike_model = RandomForestRegressor(n_estimators = 100, max_depth = 4, max_features= 0.6)
    bike_model.fit(bike_features.values, bike_targets.values.ravel())
    suicide_model = RandomForestRegressor(n_estimators = 150, max_depth = 4, max_features= 0.6)
    suicide_model.fit(suicide_features.values, suicide_targets.values.ravel())
    video_model = RandomForestRegressor(n_estimators = 100, max_depth = 4, max_features= 'auto')
    video_model.fit(video_features.values, video_targets.values.ravel())
    dot_data = tree.export_graphviz(bike_model.estimators_[0], 
                    feature_names=bike_features.columns,  
                    filled=True, rounded=True,  
                    special_characters=True,
                    out_file=None,
                    )
    graph = graphviz.Source(dot_data)
    graph.format = "png"
    graph.render("figures/q23_bike_tree.png")

    dot_data = tree.export_graphviz(suicide_model.estimators_[0], 
                feature_names=suicide_features.columns,  
                filled=True, rounded=True,  
                special_characters=True,
                out_file=None,
                )
    graph = graphviz.Source(dot_data)
    graph.format = "png"
    graph.render("figures/q23_suicide_tree.png")

    dot_data = tree.export_graphviz(video_model.estimators_[0], 
                feature_names=video_features.columns,  
                filled=True, rounded=True,  
                special_characters=True,
                out_file=None,
                )
    graph = graphviz.Source(dot_data)
    graph.format = "png"
    graph.render("figures/q23_video_tree.png")
    print("Decision Tree from Random Forest Trained on Bike Dataset")
    text_representation = tree.export_text(bike_model.estimators_[0], feature_names=list(bike_features.columns))
    print(text_representation)
    print("Decision Tree from Random Forest Trained on Suicide Dataset")
    text_representation = tree.export_text(suicide_model.estimators_[0], feature_names=list(suicide_features.columns))
    print(text_representation)
    print("Decision Tree from Random Forest Trained on Video Dataset")
    text_representation = tree.export_text(video_model.estimators_[0], feature_names=list(video_features.columns))
    print(text_representation)
    # fig = plt.figure(figsize=(25,20))
    # tree.plot_tree(bike_model.estimators_[0], feature_names=bike_features.columns, filled=True)
    # fig.savefig('figures/q23_bike_tree.png')
    # fig = plt.figure(figsize=(25,20))
    # tree.plot_tree(suicide_model.estimators_[0], feature_names=suicide_features.columns, filled=True)
    # fig.savefig('figures/q23_suicide_tree.png')
    # fig = plt.figure(figsize=(25,20))
    # tree.plot_tree(video_model.estimators_[0], feature_names=video_features.columns, filled=True)
    # fig.savefig('figures/q23_video_tree.png')

# LightGBM: https://lightgbm.readthedocs.io/en/latest/
# CatBoost: https://catboost.ai/

# LightGBM: https://lightgbm.readthedocs.io/en/latest/
# CatBoost: https://catboost.ai/
def q25():
    data_and_labels = get_all_data_and_labels_selected(pruning_for='forest')
    # TODO: pick one dataset, apply LightGBM and CatBoost.
    X, y = data_and_label=get_all_data_and_labels_selected()['bike_cnt']
    opt = BayesSearchCV(
        lgb.LGBMRegressor(),
        {
        'num_leaves': Integer(2, 50),
        'learning_rate':Real(0.01, 1, 'log-uniform'),
        'n_estimators': Integer(1, 100),
        'reg_alpha':Real(0.01, 1, 'log-uniform'),
        'subsample':Real(0.01, 1, 'log-uniform'),
        'feature_fraction':Real(0.01, 1, 'log-uniform')
            
        },
        scoring='neg_root_mean_squared_error',
        n_iter=32,
        cv=10
    )

    opt.fit(X, y)

    print("Avg. Test RMSE Error: %s" % -opt.best_score_)
    print("best_parameter: %s" % opt.best_params_)

    X, y = data_and_label=get_all_data_and_labels_selected()['bike_cnt']

    # log-uniform: understand as search over p = exp(x) by varying x
    opt = BayesSearchCV(
        CatBoostRegressor(),
        {
        'depth': Integer(1, 10),
        'learning_rate':Real(0.01, 1, 'log-uniform'),
        'n_estimators': Integer(1, 100),
        'l2_leaf_reg':Real(0.01, 1, 'log-uniform'),
        'bagging_temperature':Real(0.01, 1, 'log-uniform')     
        },
        n_iter=32,
        cv=10
    )

    opt.fit(X, y)

    print("Avg. Test RMSE Error: %s" % -opt.best_score_)
    print("best_parameter: %s" % opt.best_params_)

def q26():
    f1 = plt.figure()
    score_test=[]
    score_train=[]
    parameter=[]
    for i in np.linspace(1, 100, 25, endpoint=False):
        X, y = data_and_label=get_all_data_and_labels_selected(pruning_for='forest')['bike_cnt']
        model=lgb.LGBMRegressor(n_estimators= int(i),learning_rate=0.15,max_depth=2,num_leaves=18,reg_alpha=0.01)
        scores = cross_validate(model, X, y, scoring='neg_root_mean_squared_error', return_train_score=True, cv=10)
        score_test.append(-np.average(scores['test_score']))
        score_train.append(-np.average(scores['train_score']))
        parameter.append(i)
    plt.plot( parameter,score_test,label='Test set')
    plt.plot( parameter,score_train, label = 'Training set')
    plt.ylabel("RMSE")
    plt.xlabel("n_estimators")
    plt.title("light_gbm")
    plt.legend()
    plt.savefig(f"figures/q26_gbm_estimator.pdf")
    
    f2 = plt.figure()
    score_test=[]
    score_train=[]
    parameter=[]
    for i in np.linspace(1, 100, 25, endpoint=False):
        X, y = data_and_label=get_all_data_and_labels_selected(pruning_for='forest')['bike_cnt']
        model=lgb.LGBMRegressor(n_estimators= 100,learning_rate=0.15,max_depth=int(i),num_leaves=18,reg_alpha=0.01)
        scores = cross_validate(model, X, y, scoring='neg_root_mean_squared_error', return_train_score=True, cv=10)
        score_test.append(-np.average(scores['test_score']))
        score_train.append(-np.average(scores['train_score']))
        parameter.append(i)
    plt.plot( parameter,score_test,label='Test set')
    plt.plot( parameter,score_train, label = 'Training set')
    plt.ylabel("RMSE")
    plt.xlabel("max_depth")
    plt.title("light_gbm")
    plt.legend()
    plt.savefig(f"figures/q26_gbm_max_depth.pdf")
    
    f3 = plt.figure()
    score_test=[]
    score_train=[]
    parameter=[]
    for i in np.linspace(1, 100, 25, endpoint=False):
        X, y = data_and_label=get_all_data_and_labels_selected(pruning_for='forest')['bike_cnt']
        model=lgb.LGBMRegressor(n_estimators= 100,learning_rate=0.15,max_depth=2,num_leaves=int(i),reg_alpha=0.01)
        scores = cross_validate(model, X, y, scoring='neg_root_mean_squared_error', return_train_score=True, cv=10)
        score_test.append(-np.average(scores['test_score']))
        score_train.append(-np.average(scores['train_score']))
        parameter.append(i)
    plt.plot( parameter,score_test,label='Test set')
    plt.plot( parameter,score_train, label = 'Training set')
    plt.ylabel("RMSE")
    plt.xlabel("num_leaves")
    plt.title("light_gbm")
    plt.legend()
    plt.savefig(f"figures/q26_gbm_num_leaves.pdf")
    
    f4 = plt.figure()
    score_test=[]
    score_train=[]
    parameter=[]
    for i in np.logspace(-3, 0, 25, endpoint=False):
        X, y = data_and_label=get_all_data_and_labels_selected(pruning_for='forest')['bike_cnt']
        model=lgb.LGBMRegressor(n_estimators= 100,learning_rate=i,max_depth=2,num_leaves=18,reg_alpha=0.01)
        scores = cross_validate(model, X, y, scoring='neg_root_mean_squared_error', return_train_score=True, cv=10)
        score_test.append(-np.average(scores['test_score']))
        score_train.append(-np.average(scores['train_score']))
        parameter.append(i)
    plt.plot( parameter,score_test,label='Test set')
    plt.plot( parameter,score_train, label = 'Training set')
    plt.ylabel("RMSE")
    plt.xlabel("learning_rate")
    plt.title("light_gbm")
    plt.legend()
    plt.savefig(f"figures/q26_gbm_learning_rate.pdf")
    
    f5 = plt.figure()
    score_test=[]
    score_train=[]
    parameter=[]
    for i in np.logspace(-3, 0, 25, endpoint=False):
        X, y = data_and_label=get_all_data_and_labels_selected(pruning_for='forest')['bike_cnt']
        model=lgb.LGBMRegressor(n_estimators= 64,learning_rate=0.15,max_depth=2,num_leaves=18,reg_alpha=i)
        scores = cross_validate(model, X, y, scoring='neg_root_mean_squared_error', return_train_score=True, cv=10)
        score_test.append(-np.average(scores['test_score']))
        score_train.append(-np.average(scores['train_score']))
        parameter.append(i)
    plt.plot( parameter,score_test,label='Test set')
    plt.plot( parameter,score_train, label = 'Training set')
    plt.ylabel("RMSE")
    plt.xlabel("reg_alpha")
    plt.title("light_gbm")
    plt.legend()
    plt.savefig(f"figures/q26_gbm_reg_alpha.pdf")
    

    f6 = plt.figure()
    score_test=[]
    score_train=[]
    parameter=[]
    for i in np.linspace(1, 100, 25, endpoint=False):
        X, y = data_and_label=get_all_data_and_labels_selected(pruning_for='forest')['bike_cnt']
        model=CatBoostRegressor(n_estimators= int(i),bagging_temperature=1,depth=1, l2_leaf_reg=0.01,learning_rate=0.0824119,)
        scores = cross_validate(model, X, y, scoring='neg_root_mean_squared_error', return_train_score=True, cv=10)
        score_test.append(-np.average(scores['test_score']))
        score_train.append(-np.average(scores['train_score']))
        parameter.append(i)
    plt.plot( parameter,score_test,label='Test set')
    plt.plot( parameter,score_train, label = 'Training set')
    plt.ylabel("RMSE")
    plt.xlabel("n_estimators")
    plt.title("CatBoost")
    plt.legend()
    plt.savefig(f"figures/q26_CatBoost_n_estimators.pdf")

    f7 = plt.figure()
    score_test=[]
    score_train=[]
    parameter=[]
    for i in np.linspace(1, 100, 25, endpoint=False):
        X, y = data_and_label=get_all_data_and_labels_selected(pruning_for='forest')['bike_cnt']
        model=CatBoostRegressor(n_estimators= 64,bagging_temperature=1,depth=int(i), l2_leaf_reg=0.01,learning_rate=0.0824119,)
        scores = cross_validate(model, X, y, scoring='neg_root_mean_squared_error', return_train_score=True, cv=10)
        score_test.append(-np.average(scores['test_score']))
        score_train.append(-np.average(scores['train_score']))
        parameter.append(i)
    plt.plot( parameter,score_test,label='Test set')
    plt.plot( parameter,score_train, label = 'Training set')
    plt.ylabel("RMSE")
    plt.xlabel("depth")
    plt.title("CatBoost")
    plt.legend()
    plt.savefig(f"figures/q26_CatBoost_depth.pdf")
    
    
    f8 = plt.figure()
    score_test=[]
    score_train=[]
    parameter=[]
    for i in np.linspace(1, 100, 25, endpoint=False):
        X, y = data_and_label=get_all_data_and_labels_selected(pruning_for='forest')['bike_cnt']
        model=CatBoostRegressor(n_estimators= 64,bagging_temperature=int(i),depth=1, l2_leaf_reg=0.01,learning_rate=0.0824119,)
        scores = cross_validate(model, X, y, scoring='neg_root_mean_squared_error', return_train_score=True, cv=10)
        score_test.append(-np.average(scores['test_score']))
        score_train.append(-np.average(scores['train_score']))
        parameter.append(i)
    plt.plot( parameter,score_test,label='Test set')
    plt.plot( parameter,score_train, label = 'Training set')
    plt.ylabel("RMSE")
    plt.xlabel("bagging_temperature")
    plt.title("CatBoost")
    plt.legend()
    plt.savefig(f"figures/q26_CatBoost_bagging_temperatures.pdf")
    

    f9 = plt.figure()
    score_test=[]
    score_train=[]
    parameter=[]
    for i in np.logspace(-3, 0, 25, endpoint=False):
        X, y = data_and_label=get_all_data_and_labels_selected(pruning_for='forest')['bike_cnt']
        model=CatBoostRegressor(n_estimators= 100,learning_rate=i ,bagging_temperature=1,depth=1, l2_leaf_reg=0.01)
        scores = cross_validate(model, X, y, scoring='neg_root_mean_squared_error', return_train_score=True, cv=10)
        score_test.append(-np.average(scores['test_score']))
        score_train.append(-np.average(scores['train_score']))
        parameter.append(i)
    plt.plot( parameter,score_test,label='Test set')
    plt.plot( parameter,score_train, label = 'Training set')
    plt.ylabel("RMSE")
    plt.xlabel("learning_rate")
    plt.title("CatBoost")
    plt.savefig(f"figures/q26_CatBoost_learning_rate.pdf")
    

    
    f10 = plt.figure()
    score_test=[]
    score_train=[]
    parameter=[]
    for i in np.logspace(-3, 0, 25, endpoint=False):
        X, y = data_and_label=get_all_data_and_labels_selected(pruning_for='forest')['bike_cnt']
        model=CatBoostRegressor(n_estimators= 100,learning_rate=0.0824119 ,bagging_temperature=1,depth=1, l2_leaf_reg=i)
        scores = cross_validate(model, X, y, scoring='neg_root_mean_squared_error', return_train_score=True, cv=10)
        score_test.append(-np.average(scores['test_score']))
        score_train.append(-np.average(scores['train_score']))
        parameter.append(i)
    plt.plot( parameter,score_test,label='Test set')
    plt.plot( parameter,score_train, label = 'Training set')
    plt.ylabel("RMSE")
    plt.xlabel("l2_leaf_reg=0.01")
    plt.title("CatBoost")
    plt.savefig(f"figures/q26_CatBoost_l2_leaf_reg.pdf")
    
    f11 = plt.figure()
    score_test=[]
    score_train=[]
    parameter=[]
    for i in np.logspace(-3, 0, 25, endpoint=False):
        X, y = data_and_label=get_all_data_and_labels_selected(pruning_for='forest')['bike_cnt']
        model=lgb.LGBMRegressor(n_estimators= 64,learning_rate=i,max_depth=2,num_leaves=18,reg_alpha=0.01)
        scores = cross_validate(model, X, y, scoring='neg_root_mean_squared_error', return_train_score=True, cv=10)
        score_test.append(np.average(scores['fit_time']))
        score_train.append(np.average(scores['fit_time']))
        parameter.append(i)
    plt.plot( parameter,score_test,)
    plt.plot( parameter,score_train,)
    plt.ylabel("Fitting time")
    plt.xlabel("learning_rate")
    plt.title("light_gbm")
    plt.legend()
    plt.savefig(f"figures/q26_gbm_fitting.pdf")
    
    
    f12 = plt.figure()
    score_test=[]
    score_train=[]
    parameter=[]
    for i in np.logspace(-3, 0, 25, endpoint=False):
        X, y = data_and_label=get_all_data_and_labels_selected(pruning_for='forest')['bike_cnt']
        model=CatBoostRegressor(n_estimators= 100,learning_rate=i ,bagging_temperature=1,depth=1, l2_leaf_reg=0.01)
        scores = cross_validate(model, X, y, scoring='neg_root_mean_squared_error', return_train_score=True, cv=10)
        score_test.append(np.average(scores['fit_time']))
        score_train.append(np.average(scores['fit_time']))
        parameter.append(i)
    plt.plot( parameter,score_test,)
    plt.plot( parameter,score_train,)
    plt.ylabel("Fitting time")
    plt.xlabel("learning_rate")
    plt.title("CatBoost")
    plt.savefig(f"figures/q26_CatBoost__fitting.pdf")



if __name__=="__main__":
    q1_q2()
    q3()
    q4()
    q5()
    q5_one_plot()
    q6()
    q7()
    q8()
    q9()
    q10()
    q10_q11_13_linear()
    q10_q11_13_lasso()
    q10_q11_13_ridge()
    q10_q11_13_linear_shuffle()
    q10_q11_13_lasso_shuffle()
    q10_q11_13_ridge_shuffle()
    q12_linear()
    q12_lasso()
    q12_ridge()
    q14()
    q14_q15_16_poly()
    q17_18_19_20_NN()
    q18()
    q21_22()
    q21_22_oob()
    q21_22_r()
    q23()
    q23_plot()
    q25()
    q26()
    pass
