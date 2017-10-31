from sklearn.ensemble import RandomForestClassifier as RFC

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

col_namesfp = open('./data/selected_cols', 'r')
col_list = []
for i in range(0, 27):
    col_list.append(col_namesfp.readline().replace('\n', ''))
col_namesfp.close()

df_list = []

for year_num in range(2011, 2017):
    file_name = './data/Data' + str(year_num)+'.csv'
    df = pd.read_csv(file_name, header=0)

    # selected_df = df[['First Harmful Event of Entire Crash', 'Time Slicing Used', 'ROADWAY_ALIGNMENT', 'ROADWAY_SURFACE_COND',
    #                   'DRIVERGEN', 'CRASH_SEVERITY']].copy()

    print col_list

    selected_df = df[col_list].copy()

    selected_df = selected_df.apply(lambda x: pd.factorize(x)[0])
    # print selected_df
    df_list.append(selected_df)

selected_df = pd.concat(df_list)

# use 85% of the dataset as training set.
selected_df['is_train'] = np.random.uniform(0, 1, len(selected_df)) <= 0.85

train = selected_df[selected_df['is_train']==True]
test = selected_df[selected_df['is_train']==False]
features = selected_df.columns[0:26]

print features
# print train
# print test

forest = RFC(n_jobs=8, n_estimators=320)

#y, _ = pd.factorize(train['CRASH_SEVERITY'])
y = train['CRASH_SEVERITY']

forest.fit(train[features], y)
preds = forest.predict(test[features])
print pd.crosstab(index=test['CRASH_SEVERITY'], columns=preds, rownames=['actual'], colnames=['preds'])

importances = forest.feature_importances_
indices = np.argsort(importances)

fig_title = 'Feature Importances - All'
plt.figure(1)
plt.title(fig_title)
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.subplots_adjust(left=0.2)

plt.show()

# Appendix : raw column names
# So you can find columns you need, without battling Excel, or anything similiar
#################################################################################
# ALCOHOL_NOTALCOHOL
# A_CRASH
# A_PEOPLE
# BELTED_UNBELTED
# BIKE_NONBIKE
# BMP
# B_CRASH
# B_PEOPLE
# COLLISION_TYPE
# COMM_CARGO_BODY_TYPE_CD
# COMM_VEHICLE_BODY_TYPE_CD
# CRASH_DSC
# CRASH_DT (copy)
# CRASH_DT
# CRASH_EVENT_TYPE_DSC
# CRASH_MILITARY_TM
# CRASH_SEVERITY
# C_CRASH
# C_PEOPLE
# Area Type Used
# First Harmful Event of Entire Crash
# Physical_Juris_Name
# Time Slicing Used
# Ownership_Used
# LATITUDE
# Plan District
# Offset-Ft
# Intersection Analysis
# DAY_OF_WEEK
# DEER_NODEER
# DIRECTION_OF_TRAVEL_CD
# DISTRACTED_NOTDISTRACTED
# TOTAL CRASH
# DOCUMENT_NBR
# DRIVERAGE
# DRIVERGEN
# DRIVERINJURYTYPE
# DRIVER_ACTION_TYPE_CD
# DRIVER_DISTRACTION_TYPE_CD
# DRIVER_DRINKING_TYPE_CD
# DRIVER_SAFETY_EQUIP_USED
# FAC
# FIRST_HARMFUL_EVENT
# FUN
# GR_NOGR
# HITRUN_NOT_HITRUN
# INJURY_CRASHES
# INTERSECTION_TYPE
# K_PEOPLE
# LONGITUDE
# LGTRUCK_NONLGTRUCK
# LIGHT_CONDITION
# MAINLINE (group)
# MAINLINE_YN
# MOTOR_NONMOTOR
# Mpo Name
# NODE_INFO
# OWNERSHIP
# PASSAGE
# PASSGEN
# PASSINJURYTYPE
# PDO_CRASH
# PEDAGE
# PEDESTRIANS_INJURED
# PEDESTRIANS_KILLED
# PEDGEN
# PEDINJURYTYPE
# PED_NONPED
# PERSONS_INJURED
# PERSONS_KILLED
# RD_TYPE
# RELATION_TO_ROADWAY
# RNS_MP
# ROADWAY_ALIGNMENT
# ROADWAY_DESCRIPTION
# ROADWAY_SURFACE_COND
# ROUTE_OR_STREET_NM
# RTE_NM
# SCHOOL_ZONE
# SENIOR_NOTSENIOR
# SPEED_BEFORE
# SPEED_MAX_SAFE
# SPEED_NOTSPEED
# SPEED_POSTED
# SUMMONS_ISSUED_CD
# TIME_SLICING
# TRAFFIC_CONTROL_TYPE
# VEHICLENUMBER
# VEHICLE_BODY_TYPE_CD
# VEHICLE_MAKE_NM
# VEHICLE_MANEUVER_TYPE_CD
# VEHICLE_MODEL_NM
# VEHICLE_YEAR_NBR
# District_Used
# WEATHER_CONDITION
# WORK_ZONE_LOCATION
# WORK_ZONE_RELATED
# WORK_ZONE_TYPE
# YOUNG_NOTYOUNG

