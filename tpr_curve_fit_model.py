import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr
import operator
from python_scripts.discretize_by_MI import discretize
from python_scripts.features.feature_collection import FeatureCollection as fc
import random
import matplotlib.pyplot as plt
import scipy.integrate as integrate


def func(x, a, b, c, d):
    """reverse sigmoid function (logistic function), that doesn't drops from `c+d` to `d` with mid-point
       of `a` and drop rate of `b`.
    """
    return (c / (1 + np.exp(b*(x - a)))) + d


def min_func(params, x, y, lambda2=0.0):
    """
    Loss function to minimize.
    loss1 - asymmetrical prediction inaccuracy loss. pays more for predicting values below the true value, and much less for 
            predicting above. loss is proportional to the difference between the prediction and the actual value.
    loss2 - pays for creating a sigmoid that predicts high values (assuming input is distributed uniformly)
    """
    po = [3, 2]
    y_pred = func(x, params[0], params[1], params[2], params[3])
    loss1 = np.sum(((y > y_pred) * (y - y_pred)) ** po[0] + ((y < y_pred) * (y_pred - y) ** po[1]))

    loss2 = 0
    if lambda2 > 0:
        x0 = np.linspace(np.min(x), np.max(x), 100)
        y0 = func(x0, params[0], params[1], params[2], params[3])
        loss2 = np.sum(y0)
    
    # alternative loss2 - use the distribution of the observed x values
    # loss2 = np.sum(y_pred)
    return loss1 + (lambda2 * loss2)


def normalize_array(arr, min_val=None, max_val=None):
    if min_val is None:
        min_val = min(arr)
    if max_val is None:
        max_val = max(arr)
    return (arr-min_val)/(max_val-min_val)


def restore_normalized_array(arr, min_val, max_val):
    return (max_val-min_val)*arr+min_val


def calc_popt(x, y, direction, verbose=False):

    max_x = np.nanmax(x)
    min_x = np.nanmin(x)
    x_range = max_x - min_x
    max_y = np.nanmax(y)
    b_bound = [-100/x_range, 100/x_range]

    if direction == 'neg':
        b_bound = [0, 100/x_range]
    
    if direction == 'pos':
        b_bound = [-100/x_range, 0]

    bounds = ([min_x - 2 * x_range, max_x + 2 * x_range], b_bound, [0, 2 * max_y], [0, np.nanmax(y[x > max_x/2])])
    popts = []
    for initial_a in np.linspace(*bounds[0], num=5):
        initial = [initial_a, 0, max_y, np.nanmax(y[x > max_x/2])]
        popt = minimize(min_func, initial, args=(x, y), bounds=bounds)
        popts.append(popt)

    popt = min(popts, key=lambda x: x.fun)
    if verbose:
        print('popt', popt)
        print('pearsonr', pearsonr(x, y))

    return popt


def calc_smooth_fit(direction, neg_fit_clipped, pos_fit_clipped, full_fit, full_y_range, minimum_fitted_y_range_ratio):
    if direction != 'unknown':
        fit_clipped = neg_fit_clipped if direction == 'neg' else pos_fit_clipped
        smooth_fit = calc_popt(np.array(list(fit_clipped.keys())),
                               np.array(list(fit_clipped.values())),
                               direction=direction)
        return smooth_fit, direction, fit_clipped

    smooth_fit_neg = calc_popt(np.array(list(neg_fit_clipped.keys())),
                               np.array(list(neg_fit_clipped.values())),
                               direction='neg')
    smooth_fit_pos = calc_popt(np.array(list(pos_fit_clipped.keys())),
                               np.array(list(pos_fit_clipped.values())),
                               direction='pos')
    x_step_neg = np.fromiter(neg_fit_clipped.keys(), dtype=float)
    y_step_neg = np.fromiter(neg_fit_clipped.values(), dtype=float)
    x_step_pos = np.fromiter(pos_fit_clipped.keys(), dtype=float)
    y_step_pos = np.fromiter(pos_fit_clipped.values(), dtype=float)

    y0_neg = predict_popt(x_step_neg, smooth_fit_neg)
    y0_pos = predict_popt(x_step_pos, smooth_fit_pos)

    pos_fit_is_invalid = (np.abs(y0_pos[-1] - y0_pos[0]) / full_y_range) < minimum_fitted_y_range_ratio
    neg_fit_is_invalid = (np.abs(y0_neg[-1] - y0_neg[0]) / full_y_range) < minimum_fitted_y_range_ratio
    pos_mse = ((((y_step_pos - y0_pos) ** 2).sum()/len(y0_pos)) ** 0.5)
    neg_mse = ((((y_step_neg - y0_neg) ** 2).sum()/len(y0_neg)) ** 0.5)

    if pos_fit_is_invalid and neg_fit_is_invalid:
        return None, 'unknown', full_fit
    if pos_fit_is_invalid or (pos_mse/len(y0_pos) > neg_mse/len(y0_neg)):
        return smooth_fit_neg, 'neg', neg_fit_clipped
    return smooth_fit_pos, 'pos', pos_fit_clipped


def model_features(df, target, feature_list, max_nunique_for_category=1, clean_outliers_quantile=0.99,
                   step_fit_max_bins=100, minimum_fitted_y_range_ratio=0.05, verbose=False):
    data = df.copy()
    feature_fits = {}
    full_y_range = data[target].quantile(0.98) - data[target].quantile(0.02)
    data.loc[data[target] > data[target].quantile(clean_outliers_quantile), target] = data[target].quantile(clean_outliers_quantile)
    for att in feature_list:
        if att == target:
            continue

        if verbose:
            print(att)

        if data[att].nunique() < 2:
            if verbose:
                print('Skipping', att, 'as it is too degenerated')
            continue

        if data[att].nunique() <= max_nunique_for_category:
            fit = {
                att_val: val_data[target].quantile(clean_outliers_quantile)
                for att_val, val_data in data.groupby(att)
            }
            feature_fit = {
                'step_fit': None,
                'smooth_fit': None,
                'bar_fit': fit,
                'direction': 'ordinal',
                'fitting_score': 1,
            }
            feature_fit['y_range'] = calculate_y_range(feature_fit, data[att], data[target])
            feature_fits[att] = feature_fit
            continue

        '''
        StationMode - unknown (need to breakdown, e.g. qc is neg if target is not qc)
        '''

        direction = 'unknown'
        if any(att.startswith(a) for a in ['SwapTime', 'Attribution', 'GrError', 'LrError', 'TpError', 'Error',
                                           'Reset', 'Crash', 'Deadlock', 'AisleSafety', 'GrTeleLatency', 'FaultTime',
                                           'WaitForPath', 'RobotBlockedTime', 'GrDriftLate', 'GrMarginalDrift',
                                           'DisabledCellCount', 'NumDeadlockedRobots', 'TTR', 'Operator', 'GrReroute',
                                           'FaultTime_ground_unavailability', 'FaultTime_lift_unavailability','SuFreeSpace']):
            direction = 'neg'

        if any(att.startswith(a) for a in ['BaseUp', 'LinesToPick', 'RobotCount', 'Active', 'ToteSelectionActual']):
            direction = 'pos'

        fit = step_fit(data[att], data[target], step_fit_max_bins, clean_outliers_quantile)
        nanfit = {k: v for k, v in fit.items() if k == k}
        argmax_value = max(nanfit.items(), key=operator.itemgetter(1))[0]

        neg_fit_clipped = {key: fit[key] for key in fit.keys() if key >= argmax_value}
        pos_fit_clipped = {key: fit[key] for key in fit.keys() if key <= argmax_value}

        if len(pos_fit_clipped) == 1:
            pos_fit_clipped[max(fit.keys())] = pos_fit_clipped[argmax_value]
        if len(neg_fit_clipped) == 1:
            neg_fit_clipped[min(fit.keys())] = neg_fit_clipped[argmax_value]

        smooth_fit, direction, fit_clipped = calc_smooth_fit(direction, neg_fit_clipped, pos_fit_clipped, fit,
                                                             full_y_range, minimum_fitted_y_range_ratio)
        feature_fits[att] = {
            'step_fit': fit,
            'smooth_fit': smooth_fit,
            'bar_fit': None,
            'direction': direction,
            'max_x': argmax_value,
            'fitting_score': 0,
        }

        if feature_fits[att]['smooth_fit'] is not None:
            clipped_fit_keys = list(fit_clipped.keys())
            x_step = np.fromiter(fit_clipped.keys(), dtype=float)
            y_step = np.fromiter(fit_clipped.values(), dtype=float)
            yp = predict_popt(x_step, feature_fits[att]['smooth_fit'])
            bin_score = (max(clipped_fit_keys) - min(clipped_fit_keys)) / (data[att].max() - data[att].min())
            log_step = np.log(y_step[(y_step > 0) & (yp > 0)])
            log_pred = np.log(yp[(y_step > 0) & (yp > 0)])
            step_fit_rmsle = ((log_step - log_pred) ** 2).sum() / len(log_step)
            feature_fits[att]['fitting_score'] = bin_score * (1 / (step_fit_rmsle + 1))
            feature_fits[att]['y_range'] = calculate_y_range(feature_fits[att], data[att], data[target])
            if feature_fits[att]['y_range']/full_y_range < minimum_fitted_y_range_ratio:
                feature_fits[att]['smooth_fit'] = None
                feature_fits[att]['fitting_score'] = 0
    return feature_fits


def calculate_y_range(feature_fit, feature_data, target_data):
    p10 = np.nanpercentile(feature_data, 10)
    p90 = np.nanpercentile(feature_data, 90)
    data_for_range = feature_data[(feature_data >= p10) & (feature_data <= p90)]
    if len(data_for_range) == 0:
        data_for_range = feature_data
    try:
        ys = predict_popt(data_for_range, feature_fit['smooth_fit'])
    except Exception:
        max_x = max(data_for_range)
        min_x = min(data_for_range)
        ys = [v for k, v in feature_fit['bar_fit'].items() if (k >= min_x) and (k <= max_x)]
    return min (max(ys), max(target_data))- max(min(ys), min(target_data))


def step_fit(x, y, bins, percentile=.99):
    x_disc = discretize(x, bins=bins, range_or_quantile='quantile')
    disc_df = pd.DataFrame()
    disc_df['left'] = (x_disc.apply(lambda x: x.left)).astype(float)
    disc_df['right'] = (x_disc.apply(lambda x: x.right)).astype(float)
    disc_df['bin_width'] = disc_df.right - disc_df.left
    all_bins_sum = disc_df.drop_duplicates(subset=['left']).bin_width.sum()
    disc_df['bin_score'] = (all_bins_sum - disc_df.bin_width) / all_bins_sum
    # fillna to handle the case of a single bin (in which case - we just want to take the percentile as is
    disc_df['bin_percentile'] = percentile * (disc_df['bin_score'] / disc_df['bin_score'].max()).fillna(1)
    disc_df['y'] = np.array(y)

    fit = {
        min(x): y[x == min(x)].quantile(percentile)
    }
    for val, arr in disc_df.groupby('right'):
        fit[val] = arr.y.quantile(arr.bin_percentile.iloc[0])

    return fit


def predict_popt(x, popt):
    return func(x, *popt['x'])


def create_color_map(unique_values, seed = 200, force_basic_colors = False):
    colors = {}
    if force_basic_colors:
        basic_colors = ['blue', 'green', 'red', 'cyan', 'purple', 'orange', 'magenta', 'yellowgreen']
        i = 0
        for uv in unique_values:
            colors[uv] = basic_colors[i]
            i += 1
        return colors
    
    r = lambda: random.randint(0,255)

    for uv in unique_values:
        if uv == 'aborted':
            colors[uv] = '#D0D0D0'# 'rgb(200,200,200)'
        else:
            random.seed = seed
            colors[uv] = '#%02X%02X%02X' % (r(),r(),r())
    return colors


def draw_cdf(data, response_columns, by_column=None, bins=range(0, 10000, 5), response_bound='auto', linewidth=2, colors = None, **kwargs):
    notebook_bg_color = kwargs.get('notebook_bg_color', 'light')
    if notebook_bg_color == 'light':
        text_color = 'xkcd:black'
    elif notebook_bg_color == 'dark':
        text_color = 'xkcd:mint green'
    else:
        print('background color not supported! Use either "light" (default) or "dark", otherwise "light" is assumed...')
    ls = ["solid", "dashdot", "dotted", "dashed"]
    if len(response_columns) > 4:
        print('Supports up to 4 response columns at once!')
        return 1
    
    fig = plt.figure(figsize=(20,10))
    leg = list()
    if by_column is not None:
        by_column_vals = [str(item) for item in data[by_column].unique()]
        if colors is None:
            colors = create_color_map(by_column_vals)

        i=0
        for response_column in response_columns:
            for by_column_val in by_column_vals:
#                 try:
                plt.hist(data[data[by_column].astype(str)==by_column_val][response_column],
                         bins=bins, cumulative=True, density=True, histtype='step', linestyle=ls[i],
                         linewidth=linewidth, color=colors[by_column_val])
                if type(data.iloc[0][by_column]) == np.bool_:
                    if by_column_val == 'True':
                        leg.append(response_column+"-->"+by_column)
                    else:
                        leg.append(response_column+"-->"+'not '+by_column)
                else:
                    leg.append(response_column+"-->"+by_column_val)

            i += 1
    else:
        i=0
        for response_column in response_columns:
            plt.hist(data[response_column], bins=bins, cumulative=True, density=True,
                     histtype='step', linestyle=ls[i], linewidth=linewidth)
            leg.append(response_column)
            i += 1
    plt.title('Cumulative Distribution Function', fontsize=20, color=text_color)
    if response_bound == 'auto':
        response_bound = data[response_columns].max(axis=1).max()
    plt.xlim(0,response_bound)
    plt.ylim(0,1)
    plt.yticks(ticks=np.arange(0,1.1,0.1))
    plt.legend(leg, loc='lower right')

    plt.grid(axis='y')
    plt.xlabel('Value', fontsize=18, color=text_color)
    plt.ylabel('Probability',fontsize=18, color=text_color)
    plt.tick_params(axis='x', colors=text_color)
    plt.tick_params(axis='y', colors=text_color)


def ceil_key(d, key):
    if key in d:
        return key
    
    max_key = max(d.keys())
    if key > max_key:
        return max_key

    return min(k for k in d if k > key)


def assign_subsystem_to_feature(feature, feature_to_subsystem):
    if feature in feature_to_subsystem.keys():
        return feature_to_subsystem[feature]
    return 'not assigned'


def analyze_features(data, target, feature_fits, required=400, fit_type='smooth_fit', verbose=False):
    '''
    fit_type can be either smooth_fit (default) or step_fit
    '''

    subsystem_to_feature = fc.get_owners(data)
    feature_to_subsystem = {}
    for k, v in subsystem_to_feature.items():
        for vv in v:
            feature_to_subsystem[vv] = k

    df_filt = data.copy()
    for feat in feature_fits.keys():
        if verbose:
            print('Analyzing', feat)
        if fit_type=='smooth_fit':
            if feature_fits[feat]['smooth_fit'] is not None:
                popt = feature_fits[feat][fit_type]
                df_filt[feat+'_pred'] = predict_popt(df_filt[feat], popt)
            else:
                try:
                    df_filt[feat+'_pred'] = df_filt[feat].apply(lambda k: k if np.isnan(k) else feature_fits[feat]['step_fit'][ceil_key(feature_fits[feat]['step_fit'], k)])
                except:
                    try:
                        df_filt[feat+'_pred'] = df_filt[feat].apply(lambda k: k if np.isnan(k) else feature_fits[feat]['bar_fit'][ceil_key(feature_fits[feat]['bar_fit'], k)])
                    except:
                        df_filt[feat+'_pred'] = required
        else:
            if feature_fits[feat]['step_fit'] is not None:
                df_filt[feat+'_pred'] = df_filt[feat].apply(lambda k: k if np.isnan(k) else feature_fits[feat]['step_fit'][ceil_key(feature_fits[feat]['step_fit'], k)])
            else:
                df_filt[feat+'_pred'] = df_filt[feat].apply(lambda k: k if np.isnan(k) else feature_fits[feat]['bar_fit'][ceil_key(feature_fits[feat]['bar_fit'], k)])
        if (feature_fits[feat]['smooth_fit'] is None) and (feature_fits[feat]['bar_fit'] is None):
            df_filt[feat+'_sensitivity'] = 0
        else:
            df_filt[feat+'_sensitivity'] = (required - df_filt[feat+'_pred']).clip(lower=0)
    
    df_filt['to_explain'] = (required - df_filt[target]).clip(lower=0)
    df_filt['max_sent'] = df_filt[df_filt.columns[df_filt.columns.str.endswith('sensitivity')]].max(axis=1)
    df_filt['relevant_sent'] = df_filt[['to_explain', 'max_sent']].min(axis=1)
    df_filt['unexplained_sensitivity'] = (df_filt['to_explain'] - df_filt['max_sent']).clip(lower=0)
    total_sent = df_filt[df_filt.columns[df_filt.columns.str.endswith('sensitivity')]].sum(axis=1)
    for fs in df_filt.columns[df_filt.columns.str.endswith('sensitivity')]:
        df_filt[fs] = df_filt[fs] / (total_sent + .000000001) * (df_filt['relevant_sent'] + .000000001)
    df_filt['culprit_feature'] = df_filt[df_filt.columns[df_filt.columns.str.endswith('sensitivity')]].idxmax(axis=1)
    df_filt.loc[df_filt[df_filt.columns[df_filt.columns.str.endswith('sensitivity')]].sum(axis=1) == 0, 'culprit_feature'] = 'unexplained_sensitivity'
    df_filt['min_pred'] = df_filt[df_filt.columns[df_filt.columns.str.endswith('pred')]].min(axis=1)

    # Loop over the subsystems
    
    df_filt['unknown_subsystem'] = 0
    for sbs in subsystem_to_feature.keys():
        # Create a column and obtain the sum of sensitivities of its corresponding features
        df_filt[sbs+'_subsystem'] = 0
        for col_sensitivity in df_filt.columns[df_filt.columns.str.endswith('sensitivity')]:
            col = col_sensitivity[:-12]
            if col in feature_to_subsystem.keys():
                if feature_to_subsystem[col] == sbs:
                    df_filt[sbs+'_subsystem'] += df_filt[col_sensitivity].fillna(0)
            elif col!='unexplained':
                df_filt['unknown_subsystem'] += df_filt[col_sensitivity].fillna(0)

    df_filt['unknown_subsystem'] /= len(subsystem_to_feature.keys())
    df_filt['unexplained_subsystem'] = 0
    for col_sensitivity in df_filt.columns[df_filt.columns.str.endswith('sensitivity')]:
        col = col_sensitivity[:-12]
        if col == 'unexplained':
            df_filt['unexplained_subsystem'] += df_filt[col_sensitivity]
    subsystem_columns = df_filt.columns[df_filt.columns.str.endswith('subsystem')]
    df_filt['culprit_subsystem'] = df_filt[subsystem_columns].idxmax(axis=1)
    df_filt.loc[df_filt[subsystem_columns].sum(axis=1) == 0, 'culprit_subsystem'] = 'unexplained_subsystem'

    system_data = []
    df_filt['culprit_subsystem'] = df_filt['culprit_subsystem'].str.replace('_subsystem', '')
    for sbs in subsystem_columns.str.replace('_subsystem', ''):
        sbs_data = {
            'subsystem': sbs,
            'cumulative_loss': df_filt[sbs+'_subsystem'].sum(),
            'loss': df_filt[sbs+'_subsystem'].mean(),
            'occurrences': (df_filt[sbs + '_subsystem'] > 0).sum(),
        }
        if sbs_data['occurrences'] > 0:
            sbs_data['shadow'] = df_filt[df_filt[sbs+'_subsystem'] > 0]['culprit_subsystem'].mode().iloc[0]
        else:
            sbs_data['shadow'] = 'None'
        system_data.append(sbs_data)
    culprit_subsystems = pd.DataFrame(system_data).sort_values('loss', ascending=False)
    culprit_subsystems['loss'] = np.round(culprit_subsystems['loss'], 3)
    culprit_subsystems['cumulative_loss'] = np.round(culprit_subsystems['cumulative_loss'], 3)

    columns = []
    df_filt['culprit_feature'] = df_filt['culprit_feature'].str.replace('_sensitivity', '')
    for feat in feature_fits.keys():
        feat_data = {
            'feature': feat,
            'cumulative_loss_sensitivity': df_filt[feat+'_sensitivity'].sum(),
            'loss_sensitivity': df_filt[feat+'_sensitivity'].mean(),
            'occurrence_sensitivity': (df_filt[feat+'_sensitivity'] > 0).sum(),
            'fitting_score': feature_fits[feat]['fitting_score'],
            'y_range': feature_fits[feat].get('y_range', None),
        }
        if feat_data['occurrence_sensitivity'] > 0:
            feat_data['shadow'] = df_filt[df_filt[feat+'_sensitivity'] > 0]['culprit_feature'].mode().iloc[0]
        else:
            feat_data['shadow'] = 'None'
        columns.append(feat_data)

    culprit_features = pd.DataFrame(columns)
    culprit_features['loss_sensitivity'] = np.round(culprit_features['loss_sensitivity'], 3)
    culprit_features['cumulative_loss_sensitivity'] = np.round(culprit_features['cumulative_loss_sensitivity'], 3)
    culprit_features['fitting_score'] = np.round(culprit_features['fitting_score'], 3)
    culprit_features['subsystem'] = culprit_features['feature'].map(feature_to_subsystem)

    loss_sensitivity = culprit_features['loss_sensitivity']
    if loss_sensitivity.max() == 0:
        loss_sensitivity = 1
    occurrence_sensitivity = 1 + culprit_features['occurrence_sensitivity']
    fitting_score = culprit_features['fitting_score']
    y_range_score = (culprit_features['y_range'].fillna(0).abs() + 1) / \
                    (data[target].quantile(0.98) - data[target].quantile(0.02))
    culprit_features['total_score'] = loss_sensitivity * occurrence_sensitivity * fitting_score * y_range_score

    culprit_features['direction'] = culprit_features.apply(lambda x: feature_fits[x['feature']]['direction'], axis=1)
    return culprit_features, culprit_subsystems, df_filt


def plot_subsystem_breakdown(subsystems, culprit_features, top_k=10, by='total_score', **kwargs):
    notebook_bg_color = kwargs.get('notebook_bg_color', 'light')
    if notebook_bg_color == 'light':
        text_color = 'xkcd:black'
    elif notebook_bg_color == 'dark':
        text_color = 'xkcd:mint green'
    else:
        print('background color not supported! Use either "light" (default) or "dark", otherwise "light" is assumed...')
    if type(subsystems) != list:
        subsystems = [subsystems]
    df = culprit_features[culprit_features['subsystem'].isin(subsystems)]
    mini = min(top_k, df.shape[0])
    if mini == 0:
        return
    df_to_draw = df.sort_values(by, ascending=False).head(mini).sort_values(by)
    ax = df_to_draw.plot.barh(x='feature', y='loss_sensitivity', figsize=(20,10))
    plt.title('Feature breakdown for '+str(subsystems)+' subsystem', color=text_color, fontsize=20)
    plt.xlabel('TPR Potential Gain', fontsize=18, color=text_color)
    plt.ylabel('Feature',fontsize=18, color=text_color)
    plt.tick_params(axis='x', colors=text_color)
    plt.tick_params(axis='y', colors=text_color)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    plt.show()


def draw_feature_fits(data, target, feature_fits, culprit_features, normalized=True, columns=4, figsize=(8, 8), **kwargs):
    notebook_bg_color = kwargs.get('notebook_bg_color', 'light')
    show_step_fit = kwargs.get('show_step_fit', False)
    feature_fits_union = kwargs.get('feature_fits_union', None)
    fill_between = kwargs.get('fill_between', False)
    scatter_patterns = kwargs.get('scatter_patterns', ['ro', 'bo'])
    if notebook_bg_color == 'light':
        text_color = 'xkcd:black'
    elif notebook_bg_color == 'dark':
        text_color = 'xkcd:mint green'
    else:
        print('background color not supported! Use either "light" (default) or "dark", otherwise "light" is assumed...')

    if type(data) != list:
        data = [data]
    
    if type(feature_fits) != list:
        feature_fits = [feature_fits]
    
    if type(culprit_features) != list:
        culprit_features = [culprit_features]

    
    fig, axs = plt.subplots(len(feature_fits[0].keys())//columns+1, columns, figsize=figsize)
    fig.suptitle(target, color=text_color, y=1.005)
    fig.tight_layout(h_pad=2)

    if feature_fits_union is not None:
        data_union = pd.DataFrame()
        for j in range(len(data)):
            data_union = data_union.append(data[j])
    
    i = 1
    for att in list(culprit_features[0]['feature']):
        min_x = np.inf
        max_x = -np.inf
        min_y = np.inf
        max_y = -np.inf

        if feature_fits_union is not None:
            if att not in list(feature_fits_union.keys()):
                feature_fits_union[att] = {'step_fit': None, 'smooth_fit': None, 'bar_fit': None}

        x = [None, None]
        y = [None, None]
        for j in range(len(data)):
            if att not in list(feature_fits[j].keys()):
                feature_fits[j][att] = {'step_fit': None, 'smooth_fit': None, 'bar_fit': None}

            x[j] = data[j][att]
            y[j] = data[j][target]

            if feature_fits[j][att]['step_fit'] is not None:
                min_x = min(min_x, min(feature_fits[j][att]['step_fit'].keys()))
                max_x = max(max_x, max(feature_fits[j][att]['step_fit'].keys()))
                min_y = min(min_y, min(feature_fits[j][att]['step_fit'].values()))
                max_y = max(max_y, max(feature_fits[j][att]['step_fit'].values()))
            else:
                min_x = min(min_x, np.nanmin(x[j]))
                max_x = max(max_x, np.nanmax(x[j]))
                min_y = min(min_y, np.nanmin(y[j]))
                max_y = max(max_y, np.nanmax(y[j]))

            if normalized:
                if feature_fits[j][att]['smooth_fit'] is not None: 
                    x[j] = normalize_array(x[j], min(x[j]), max(x[j]))
                y[j] = normalize_array(y[j], min(y[j]), max(y[j]))
        plt.subplot(len(feature_fits[0].keys())//columns+1, columns, i, facecolor='xkcd:mint green')
        for j in range(len(data)):
            plt.plot(x[j], y[j], scatter_patterns[j], alpha=0.3)
        if len(att) > 30:
                rot = -2
                fz = 10
        else:
                rot = 0
                fz = 10
        plt.xlabel(att, rotation=rot, fontsize=fz, fontweight='bold', color=text_color)
        plt.grid(True)
        x_offset = 0.55
        if feature_fits[0][att]['smooth_fit'] is not None:
            if feature_fits[0][att]['max_x'] > (max_x - min_x) / 2:
                x_offset = 0.05
        text_x = min_x + x_offset*(max_x - min_x)
        if len(data) == 1:
            plt.text(text_x, .9*max(y[0]), 'occ='+str(culprit_features[0][culprit_features[0]['feature']==att]['occurrence_sensitivity'].iloc[0]))
            plt.text(text_x, .75*max(y[0]), 'loss='+str(np.round(culprit_features[0][culprit_features[0]['feature']==att]['loss_sensitivity'].iloc[0],2)))
            if culprit_features[0][culprit_features[0]['feature']==att]['fitting_score'] is None:
                plt.text(text_x, .6*max(y[0]), 'fit=n/a')
            else:
                plt.text(text_x, .6*max(y[0]), 'fit='+str(np.round(culprit_features[0][culprit_features[0]['feature']==att]['fitting_score'].iloc[0],3)))
            plt.text(text_x, .95*max(y[0]), 'total score='+str(np.round(culprit_features[0][culprit_features[0]['feature']==att]['total_score'].iloc[0],2)))
            shadowing = culprit_features[0][culprit_features[0]['feature']==att]['shadow'].iloc[0]
            if (shadowing != att) & (shadowing != 'unexplained'):
                plt.text(min_x+0.1*(max_x - min_x), .01+min_y, 'shadow of '+shadowing)
            if feature_fits[0][att]['step_fit'] is not None:
                fit_type = 'step_fit'       
            if feature_fits[0][att]['bar_fit'] is not None:
                fit_type = 'bar_fit'
            A = feature_fits[0][att][fit_type]
            sorted_dict = {k:A[k] for k in sorted(A)}
            x0 = list(sorted_dict.keys())
            y0 = list(sorted_dict.values())
            if normalized:
                y0 = normalize_array(np.array(y0), np.nanmin(data[0][target]), np.nanmax(data[0][target]))
            if show_step_fit:
                plt.plot(x0, y0, 'k:', drawstyle='steps', linewidth=1)
        x0 = [None, None]
        y0 = [None, None]
        for j in range(len(data)): 
            if feature_fits[j][att]['smooth_fit'] is not None:
                x0[j], y0[j] = _obtain_x_y(feature_fits[j], target, att, data[j], min_x, max_x, normalized)
                if j==0:
                    curve_pattern = 'r-'
                else:
                    curve_pattern = 'b-'
                plt.plot(x0[j], y0[j], curve_pattern, lw=3)

        if fill_between:
            try:
                fill_from = max([min(x0[0]), min(x0[1])])
                fill_to = min([max(x0[0]), max(x0[1])])
                x0 = np.linspace(fill_from, fill_to , 100)
                for j in range(len(data)):
                    y0[j] = predict_popt(x0, feature_fits[j][att]['smooth_fit'])
                plt.fill_between(x0, y0[0], y0[1], alpha=0.5)
                first_area = integrate.quad(lambda x: predict_popt(x, feature_fits[0][att]['smooth_fit']), fill_from, fill_to)[0]
                second_area = integrate.quad(lambda x: predict_popt(x, feature_fits[1][att]['smooth_fit']), fill_from, fill_to)[0]
                plt.text(text_x, 1.1*min(y0[0]), 'diff='+str(np.round((second_area - first_area)/(fill_to - fill_from),2)))
            except:
                print('Cant fill between for', att)

        if feature_fits_union is not None:
            x0, y0 = _obtain_x_y(feature_fits_union, target, att, data_union, min_x, max_x, normalized)
            if (x0 is not None) & (y0 is not None):
                plt.plot(x0, y0, 'm--', lw=3)

        plt.tick_params(axis='x', colors=text_color)
        plt.tick_params(axis='y', colors=text_color)
        i += 1
    return fig

def _obtain_x_y(feature_fits, target, att, data, min_x, max_x, normalized):
    if feature_fits[att]['smooth_fit'] is not None:
        if feature_fits[att]['direction']=='pos':
            x0 = np.linspace(min_x, feature_fits[att]['max_x'], 100)
        if feature_fits[att]['direction']=='neg':
            x0 = np.linspace(feature_fits[att]['max_x'], max_x , 100)
        if feature_fits[att]['direction']=='unknown':
            x0 = np.linspace(min_x, max_x , 100)
        if normalized:
            y0 = list(predict_popt(restore_normalized_array(x0, np.nanmin(data[att]), np.nanmax(data[att])), feature_fits[att]['smooth_fit']))
            y0 = normalize_array(np.array(y0), np.nanmin(data[target]), np.nanmax(data[target]))
        else:
            y0 = predict_popt(x0, feature_fits[att]['smooth_fit'])
        return x0, y0
    return None, None

def draw_subsystem_fits(subsystems, top_k=10, **kwargs):
    show_step_fit = kwargs.get('show_step_fit', False)
    notebook_bg_color = kwargs.get('notebook_bg_color', 'light')
    if notebook_bg_color == 'light':
        text_color = 'xkcd:black'
    elif notebook_bg_color == 'dark':
        text_color = 'xkcd:mint green'
    else:
        print('background color not supported! Use either "light" (default) or "dark", otherwise "light" is assumed...')
    data = kwargs.get('data', None)
    culprit_features = kwargs.get('culprit_features', None)
    feature_fits = kwargs.get('feature_fits', None)
    
    by_col = kwargs.get('by_col', 'total_score')
    target = kwargs.get('target', 'TPR_picking_in_and_out')
    cols = kwargs.get('cols', 4)
    if type(subsystems) != list:
        subsystems = [subsystems]
    df = culprit_features[culprit_features['subsystem'].isin(subsystems)]
    mini = min(top_k, df.shape[0])
    if mini == 0:
        return
    keys_to_extract = list(df.sort_values(by_col, ascending=False)['feature'].head(mini))
    a_subset = {key: feature_fits[key] for key in keys_to_extract}
    
    fig = draw_feature_fits(data, target, a_subset, culprit_features[culprit_features['feature'] \
        .isin(keys_to_extract)].sort_values(by_col, ascending=False), \
            columns=cols, normalized=False, figsize=(16, (16/cols) * np.ceil(len(a_subset)/cols)), notebook_bg_color=notebook_bg_color, show_step_fit=show_step_fit)

def compare_two_modes(data, target, split_by, required, **kwargs):
    '''
    Usage example:
    context = 'night_wave'
    mode = 'start'
    df[context] = (df['START_TIME'].dt.hour >= 2) & (df['START_TIME'].dt.hour <= 4)
    df_context = df[df[context]]
    df[context+'_'+mode] = df_context['START_TIME'].dt.hour <= 3
    df[context+'_rest'] = ~df_context[context+'_'+mode]
    target = 'TPR_picking_in_and_out'
    compare_two_modes(data=df_context, target=target, split_by=context+'_'+mode, required=df_context['required'])
    Note that the direction of the comparison might matter. That said, consider
    compare_two_modes(data=df_context, split_by=context+'_rest', required=df_context['required])
    as the order of the sensitivity is determined by the split_by column
    '''
    selected_features = kwargs.get('selected_features', None)
    top_k = kwargs.get('top_k', 10)
    by_col = kwargs.get('by_col', 'total_score')
    how_many_displaying_in_a_row = kwargs.get('how_many_displaying_in_a_row', 3)
    notebook_bg_color = kwargs.get('notebook_bg_color', 'light')
    plot_union = kwargs.get('plot_union', False)
    clean_outliers_quantile = kwargs.get('clean_outliers_quantile', 0.999)
    fill_between = kwargs.get('fill_between', True)
    if selected_features is None:
        selected_features = list(data.select_dtypes(include=[np.number]).columns.values)
    print('modelling...')
    model_kwargs = {
        'target': target,
        'feature_list': selected_features,
        'clean_outliers_quantile': clean_outliers_quantile
    }
    feature_fits_mode = model_features(data[data[split_by]], **model_kwargs)
    feature_fits_rest = model_features(data[~data[split_by]], **model_kwargs)
    if plot_union:
        feature_fits_union = model_features(data, **model_kwargs)
    else:
        feature_fits_union = None
    print('analyzing...')
    culprit_features_mode, culprit_subsystems_mode, df_filt_analaized_mode = analyze_features(data[data[split_by]], target, feature_fits_mode, required)
    culprit_features_rest, culprit_subsystems_rest, df_filt_analaized_rest = analyze_features(data[~data[split_by]], target, feature_fits_rest, required)
    print('drawing...')
    mini = min(top_k, culprit_features_mode.shape[0])
    if mini == 0:
        return
    keys_to_extract = list(culprit_features_mode.sort_values(by_col, ascending=False)['feature'].head(mini))
    a_subset = {key: feature_fits_mode[key] for key in keys_to_extract}
    culprit_features_mode_sub = culprit_features_mode[culprit_features_mode['feature'].isin(keys_to_extract)].sort_values(by_col, ascending=False)
    culprit_features_rest_sub = culprit_features_rest[culprit_features_rest['feature'].isin(keys_to_extract)].sort_values(by_col, ascending=False)
    fig = draw_feature_fits([df_filt_analaized_mode, df_filt_analaized_rest], target, \
                            [a_subset, feature_fits_rest], [culprit_features_mode_sub, culprit_features_rest_sub], \
            columns=how_many_displaying_in_a_row, normalized=False, figsize=(16, (16/how_many_displaying_in_a_row) \
                * np.ceil(len(a_subset)/how_many_displaying_in_a_row)), notebook_bg_color=notebook_bg_color, feature_fits_union=feature_fits_union, fill_between=fill_between)
    
def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def kl_div_from_samples(series1, series2, resolution=10):
    a = discretize(series1, resolution, 'quantile').value_counts(normalize=True)
    b = discretize(series2, resolution, 'quantile').value_counts(normalize=True)
    return kl_divergence(a.values, b.values)