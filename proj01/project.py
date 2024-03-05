# project.py


import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_assignment_names(grades):
    column_names = grades.columns.values
    final_dict = {}
    parts = ['lab', 'project', 'Midterm', 'Final', 'discussion', 'checkpoint']
    k = ['lab', 'project', 'midterm', 'final', 'disc', 'checkpoint']

    for i in np.arange(len(parts)):
        if parts[i] == 'project':
            assignments = list(filter(lambda x: parts[i] in x and '-' not in x and 'checkpoint' not in x, column_names))
        else:
            assignments = list(filter(lambda x: parts[i] in x and '-' not in x, column_names))
        
        final_dict[k[i]] = assignments

    return final_dict

# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def projects_total(grades):
    grades = grades.fillna(0)
    projects = get_assignment_names(grades)['project']
    max_pt_names = [name + ' - Max Points' for name in projects]
    max_pt = grades[max_pt_names].iloc[0]
    max_pt = pd.Series(max_pt.values, index=projects)

    projects = sorted(projects)

    project_grades = pd.DataFrame()

    for i in np.arange(len(projects) - 1):
        if projects[i] in projects[i+1]:
            project_grade = (grades[projects[i]] + grades[projects[i+1]]) / (max_pt[projects[i]] + max_pt[projects[i+1]])
            project_grades[projects[i]] = project_grade
        elif 'free_response' not in projects[i]:
            project_grade = grades[projects[i]] / max_pt[projects[i]]
            project_grades[projects[i]] = project_grade
            
    return project_grades.mean(axis=1)

# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def lateness_penalty(col):
    def multiplier(time):
        times = time.split(':')
        hours = int(times[0])
        minutes = int(times[1])
        seconds = int(times[2])
        total_time = hours + minutes/60 + seconds/360
        
        if total_time <= 2:
            return 1
        elif total_time <= 7*24:
            return 0.9
        elif total_time <= 7*24*2:
            return 0.7
        else:
            return 0.4
    
    return col.apply(multiplier)


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def process_labs(grades):
    grades = grades.fillna(0)
    labs = get_assignment_names(grades)['lab']
    max_pt_names = [name + ' - Max Points' for name in labs]
    max_pt = grades[max_pt_names].iloc[0]
    max_pt = pd.Series(max_pt.values, index=labs)

    lateness_names = [name + ' - Lateness (H:M:S)' for name in labs]

    lab_grades = pd.DataFrame()
    lateness_df = pd.DataFrame()

    for i in np.arange(len(labs)):
        lab_i = np.array(grades[labs[i]])
        lab_grades[labs[i]] = lab_i / max_pt[labs[i]]
        lateness_df[labs[i]] = lateness_penalty(grades[lateness_names[i]])
        
    return lateness_df * lab_grades

# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def lab_total(processed):

    labs = processed.columns.values
    processed['min_scores'] = processed.min(axis=1)

    lab_sum = np.zeros(processed.shape[0])
    for i in np.arange(len(labs)):
        name = labs[i]
        lab_sum = lab_sum + processed[name]

    total_grade = (lab_sum - processed['min_scores']) / (processed.shape[1] - 2)

    return total_grade


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def total_points(grades):
    grades = grades.fillna(0)

    def part_grades(part):
        assignments = get_assignment_names(grades)[part]
        max_pt_names = [name + ' - Max Points' for name in assignments]
        max_pt = grades[max_pt_names].iloc[0]
        max_pt = pd.Series(max_pt.values, index=assignments)
        
        ass_grades = pd.DataFrame()
        
        for i in np.arange(len(assignments)):
            name = assignments[i]
            assignment_i = np.array(grades[name])
            ass_grades[name] = assignment_i / max_pt[name]
        
        return ass_grades.mean(axis=1)

    all_grades = pd.DataFrame()
    parts_left = ['disc', 'checkpoint', 'final', 'midterm']

    for part in parts_left:
        all_grades[part] = part_grades(part)
        
    all_grades['project'] = projects_total(grades)
    all_grades['lab'] = lab_total(process_labs(grades))

    final_grade = (all_grades['disc'] * 0.025 +
                   all_grades['checkpoint'] * 0.025 +
                   all_grades['midterm'] * 0.15 +
                   all_grades['final'] * 0.3 +
                   all_grades['project'] * 0.3 +
                   all_grades['lab'] * 0.2)
    
    return final_grade

# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def final_grades(total):
    def grade_calc(fin_grade):
        if fin_grade >= 0.9:
            return 'A'
        elif fin_grade >= 0.8:
            return 'B'
        elif fin_grade >= 0.7:
            return 'C'
        elif fin_grade >= 0.6:
            return 'D'
        else:
            return 'F'

    return total.apply(grade_calc)

def letter_proportions(total):
    letter_grades = final_grades(total)
    return letter_grades.value_counts() / len(letter_grades)

# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def raw_redemption(redemption_scores, question_list):
    redemption_scores = redemption_scores.fillna(0)
    max_points_per_question = redemption_scores.max().drop("PID")
    total_score = redemption_scores[["PID"]]
    total_score = total_score.assign(Raw_Redemption_Score = 0)
    total_max = 0
    for i in np.arange(len(question_list)):
        question_score = redemption_scores.iloc[:,question_list[i]]
        total_score["Raw_Redemption_Score"] = total_score["Raw_Redemption_Score"] + question_score
        question_max = question_score.max()
        total_max += question_max
    total_score["Raw_Redemption_Score"] = total_score["Raw_Redemption_Score"] / total_max
    total_score = total_score.rename(columns={"Raw_Redemption_Score": "Raw Redemption Score"})
    return total_score
    
def combine_grades(full_grades, redemption_scores):
    return full_grades.merge(redemption_scores, on="PID")


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def z_score(score_series):
    mean_score = score_series.mean()
    std_score = score_series.std(ddof=0)
    return ((score_series - mean_score) / std_score)
    
def add_post_redemption(scores_with_redemption):
    scores_with_redemption['Midterm Score Pre-Redemption'] = (scores_with_redemption["Midterm"] /
                                                              scores_with_redemption['Midterm - Max Points'])
    
    redemption_zscore = z_score(scores_with_redemption['Raw Redemption Score'])
    midterm_zscore = z_score(scores_with_redemption["Midterm Score Pre-Redemption"])
    
    midterm_scores = scores_with_redemption['Midterm Score Pre-Redemption']
    
    tf_series = redemption_zscore < midterm_zscore
    
    final_score = midterm_scores.where(tf_series, (redemption_zscore * scores_with_redemption['Midterm Score Pre-Redemption'].std(ddof=0)
                           + scores_with_redemption['Midterm Score Pre-Redemption'].mean()))
    
    scores_with_redemption['Midterm Score Post-Redemption'] = final_score
    return scores_with_redemption


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def total_points_post_redemption(scores):
    scores = scores.fillna(0)

    def part_grades(part):
        assignments = get_assignment_names(scores)[part]
        max_pt_names = [name + ' - Max Points' for name in assignments]
        max_pt = scores[max_pt_names].iloc[0]
        max_pt = pd.Series(max_pt.values, index=assignments)
        
        ass_grades = pd.DataFrame()
        
        for i in np.arange(len(assignments)):
            name = assignments[i]
            assignment_i = np.array(scores[name])
            ass_grades[name] = assignment_i / max_pt[name]
        
        return ass_grades.mean(axis=1)

    all_grades = pd.DataFrame()
    parts_left = ['disc', 'checkpoint', 'final']

    for part in parts_left:
        all_grades[part] = part_grades(part)
        
    all_grades['project'] = projects_total(scores)
    all_grades['lab'] = lab_total(process_labs(scores))
    all_grades['midterm'] = scores["Midterm Score Post-Redemption"]

    final_grade = (all_grades['disc'] * 0.025 +
                   all_grades['checkpoint'] * 0.025 +
                   all_grades['midterm'] * 0.15 +
                   all_grades['final'] * 0.3 +
                   all_grades['project'] * 0.3 +
                   all_grades['lab'] * 0.2)
    return final_grade
        
def proportion_improved(scores):
    pre_redemption = final_grades(total_points(scores))
    post_redemption = final_grades(total_points_post_redemption(scores))
    mean_letter_inc = (pre_redemption > post_redemption).mean()
    return mean_letter_inc


# ---------------------------------------------------------------------
# QUESTION 11
# ---------------------------------------------------------------------

def section_most_improved(final_grades):
    final_grades_copy = final_grades.copy()
    final_grades_copy['improved'] = final_grades_copy['Letter Grade Pre-Redemption'] > final_grades_copy['Letter Grade Post-Redemption']
    best_improved = final_grades_copy.groupby('Section').mean()['improved'].idxmax()
    return best_improved
    
def top_sections(final_grades, t, n):
    final_grades_copy = final_grades.copy()
    final_grades_copy['final_percent'] = final_grades_copy['Final'] / final_grades_copy['Final - Max Points']
    final_grades_copy['raw_score_good'] = final_grades_copy['final_percent'] >= t
    total_good = final_grades_copy.groupby("Section").sum()['raw_score_good']
    total_good = total_good[total_good > n]
    return np.array(total_good.index)
    


# ---------------------------------------------------------------------
# QUESTION 12
# ---------------------------------------------------------------------


def rank_by_section(final_grades):
    agg_student_section = final_grades.groupby(['Section', 'PID']).agg({'Total Points Post-Redemption':sum})
    grouped_student_section = agg_student_section['Total Points Post-Redemption'].groupby('Section', group_keys=False)
    final_student_section = grouped_student_section.apply(lambda x: x.sort_values(ascending = False))
    final_student_section = final_student_section.to_frame()
    
    total_students = final_grades['Section'].value_counts().sort_index()
    section_rank = np.array([])
    for i in np.arange(len(total_students)):
        ranks = np.arange(1, total_students.iloc[i] + 1)
        section_rank = np.append(section_rank, ranks)
    
    final_student_section['Section Rank'] = section_rank.astype(int)
    final_student_section = final_student_section.reset_index()

    return final_student_section.pivot(index='Section Rank', columns = 'Section', values=['PID']).fillna("")

# ---------------------------------------------------------------------
# QUESTION 13
# ---------------------------------------------------------------------


def letter_grade_heat_map(final_grades):
    grade_pivot = final_grades.pivot_table(index = 'Letter Grade Post-Redemption', columns='Section', values='PID', aggfunc='count', fill_value=0)
    grade_pivot = grade_pivot / grade_pivot.sum(axis=0)
    
    fig = px.imshow(grade_pivot, color_continuous_scale = 'blugrn', title='Distribution of Letter Grades by Section')
    
    return fig
