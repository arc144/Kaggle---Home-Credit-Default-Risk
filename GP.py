import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def Output(p):
    return 1. / (1. + np.exp(-p))


def GP1(data):
    v = pd.DataFrame()
    v["i0"] = 0.059995 * np.tanh(((((((((np.maximum(((((data["REFUSED_CNT_PAYMENT_SUM"]) + (1.570796)))), ((-1.0)))) + (
        ((data["DAYS_BIRTH"]) * 2.0)))) + (((data["REGION_RATING_CLIENT"]) + (data["NAME_INCOME_TYPE_Working"]))))) * 2.0)) * 2.0))
    v["i1"] = 0.019540 * np.tanh((((13.69431781768798828)) * (((data["REGION_RATING_CLIENT_W_CITY"]) + (np.tanh((np.maximum(((data["REGION_RATING_CLIENT_W_CITY"])), ((
        (((1.0) + (((np.minimum(((data["REFUSED_DAYS_DECISION_MAX"])), ((data["REFUSED_CNT_PAYMENT_SUM"])))) + (data["REFUSED_DAYS_DECISION_MAX"])))) / 2.0)))))))))))
    v["i2"] = 0.099540 * np.tanh(((data["REGION_RATING_CLIENT_W_CITY"]) + (((((((data["REGION_RATING_CLIENT_W_CITY"]) - (np.tanh((data["CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN"]))))
                                                                               ) + (((((((data["DAYS_BIRTH"]) - (data["NAME_EDUCATION_TYPE_Higher_education"]))) * 2.0)) - (data["CODE_GENDER"]))))) * 2.0))))
    v["i3"] = 0.097305 * np.tanh(((((((np.maximum(((data["REFUSED_CNT_PAYMENT_SUM"])), ((((np.where(data["REFUSED_DAYS_DECISION_MEAN"] > 0, np.where(
        data["REFUSED_DAYS_DECISION_MAX"] > 0, 2.0, data["DAYS_BIRTH"]), data["DAYS_BIRTH"])) - (0.636620)))))) * 2.0)) * 2.0)) * 2.0))
    v["i4"] = 0.099670 * np.tanh(((((((((((np.tanh((((((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) * 2.0)) - (
        data["CLOSED_AMT_CREDIT_SUM_MEAN"]))) * 2.0)))) - (data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]))) * 2.0)) * 2.0)) * 2.0)) - (1.570796)))
    v["i5"] = 0.099503 * np.tanh(((((((np.where(data["REFUSED_DAYS_DECISION_MAX"] < 0, data["REGION_RATING_CLIENT_W_CITY"], 2.0)) + (
        ((((data["FLAG_DOCUMENT_3"]) + (((data["DAYS_ID_PUBLISH"]) + (data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]))))) - (data["CODE_GENDER"]))))) * 2.0)) * 2.0))
    v["i6"] = 0.099849 * np.tanh(((data["NEW_SCORES_STD"]) + (((((((data["REGION_RATING_CLIENT_W_CITY"]) + (((data["NAME_INCOME_TYPE_Working"]) + (
        ((data["NEW_SCORES_STD"]) + (((data["REG_CITY_NOT_WORK_CITY"]) - (data["NAME_EDUCATION_TYPE_Higher_education"]))))))))) - (data["CODE_GENDER"]))) * 2.0))))
    v["i7"] = 0.099890 * np.tanh(np.where((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) + (0.318310)) / 2.0) > 0, (5.78559446334838867), np.where(data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"]
                                                                                                                                                 < -99998, data["NEW_DOC_IND_KURT"], ((((((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) + (data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"]))) * 2.0)) * 2.0)) * 2.0))))
    v["i8"] = 0.099495 * np.tanh(np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"] > 0, (5.0), np.where(data["NEW_SCORES_STD"] > 0, data["NEW_SCORES_STD"], np.where(
        data["NEW_SCORES_STD"] < 0, ((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) + (data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"]))) * ((8.84948539733886719))), (13.10789012908935547)))))
    v["i9"] = 0.099966 * np.tanh(((np.where(((((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) * 2.0)) + (0.318310)) > 0, 3.0, ((data["REFUSED_DAYS_DECISION_MAX"]) + (
        ((((data["REFUSED_DAYS_DECISION_MEAN"]) + (data["REFUSED_CNT_PAYMENT_SUM"]))) - (data["REFUSED_HOUR_APPR_PROCESS_START_MEAN"])))))) * 2.0))
    v["i10"] = 0.099804 * np.tanh((((((((data["DAYS_ID_PUBLISH"]) + (data["REGION_RATING_CLIENT_W_CITY"]))) + (((((data["REG_CITY_NOT_LIVE_CITY"]) - (
        np.where(data["NEW_CAR_TO_BIRTH_RATIO"] > 0, 3.141593, np.tanh((data["CLOSED_DAYS_CREDIT_VAR"])))))) * 2.0))) / 2.0)) * 2.0))
    v["i11"] = 0.099970 * np.tanh(((np.where(np.maximum(((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"])), ((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]))) > 0, (6.61570215225219727), ((
        np.where(data["NAME_EDUCATION_TYPE_Higher_education"] > 0, ((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) * 2.0), data["DAYS_BIRTH"])) - (data["CODE_GENDER"])))) * 2.0))
    v["i12"] = 0.099924 * np.tanh((-1.0 * ((np.where(np.minimum(((data["NAME_FAMILY_STATUS_Married"])), ((data["NEW_CAR_TO_EMPLOY_RATIO"])))
                                                     > 0, (10.0), (((6.0)) * (np.where(data["NEW_DOC_IND_KURT"] < 0, 3.0, ((data["NEW_DOC_IND_KURT"]) + (data["EXT_SOURCE_1"]))))))))))
    v["i13"] = 0.099965 * np.tanh(((((3.0) * (((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) - (np.minimum(((np.maximum(((data["FLOORSMAX_AVG"])), ((
        data["ENTRANCES_MEDI"]))))), ((((((((data["FLOORSMAX_AVG"]) / 2.0)) / 2.0)) - (data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"])))))))))) * 2.0))
    v["i14"] = 0.099979 * np.tanh(np.where(((data["ACTIVE_DAYS_CREDIT_VAR"]) - (data["CLOSED_CNT_CREDIT_PROLONG_SUM"])) < 0, ((((data["ACTIVE_DAYS_CREDIT_VAR"]) - (
        data["CLOSED_AMT_CREDIT_SUM_MEAN"]))) + (3.0)), (((((-1.0 * ((data["CLOSED_AMT_CREDIT_SUM_MEAN"])))) * 2.0)) - (data["ACTIVE_DAYS_CREDIT_VAR"]))))
    v["i15"] = 0.099860 * np.tanh(((np.where(np.maximum(((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"])), ((((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]) + ((((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]) > (
        data["APPROVED_AMT_DOWN_PAYMENT_MEAN"])) * 1.)))))) > 0, data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"], (-1.0 * ((data["NAME_FAMILY_STATUS_Married"]))))) - (((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]) * 2.0))))
    v["i16"] = 0.097603 * np.tanh(((((data["DAYS_REGISTRATION"]) + (((data["REGION_RATING_CLIENT"]) - (data["NAME_INCOME_TYPE_State_servant"]))))) - (np.maximum(
        ((((data["CODE_GENDER"]) - (data["OCCUPATION_TYPE_Laborers"])))), ((((((data["OCCUPATION_TYPE_Core_staff"]) * 2.0)) - (data["REGION_RATING_CLIENT_W_CITY"]))))))))
    v["i17"] = 0.100000 * np.tanh(np.maximum(((((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) * 2.0)) * 2.0))), (((((data["ACTIVE_DAYS_CREDIT_VAR"]) + (
        ((((data["CC_CNT_DRAWINGS_POS_CURRENT_MEAN"]) * ((((np.tanh((data["ACTIVE_DAYS_CREDIT_VAR"]))) > (data["FLAG_DOCUMENT_3"])) * 1.)))) - (data["NEW_CAR_TO_BIRTH_RATIO"])))) / 2.0)))))
    v["i18"] = 0.099955 * np.tanh(np.where(((data["EXT_SOURCE_1"]) + (((data["CLOSED_DAYS_CREDIT_VAR"]) * (data["NAME_FAMILY_STATUS_Married"])))) < -99998, data[
                                  "ORGANIZATION_TYPE_Self_employed"], ((data["ORGANIZATION_TYPE_Self_employed"]) - ((((5.0)) * (((1.0) + (data["EXT_SOURCE_1"]))))))))
    v["i19"] = 0.099900 * np.tanh(((np.where(data["NEW_CREDIT_TO_INCOME_RATIO"] < 0, data["NEW_CREDIT_TO_INCOME_RATIO"], data["OCCUPATION_TYPE_Drivers"])) + (
        (((((((data["ORGANIZATION_TYPE_Construction"]) + (data["ORGANIZATION_TYPE_Business_Entity_Type_3"])) / 2.0)) + (data["OCCUPATION_TYPE_Drivers"]))) + ((((data["FLAG_PHONE"]) < (data["NEW_CREDIT_TO_INCOME_RATIO"])) * 1.))))))
    v["i20"] = 0.099957 * np.tanh(((np.tanh((data["REFUSED_CNT_PAYMENT_SUM"]))) + (np.maximum((((((data["OCCUPATION_TYPE_Low_skill_Laborers"]) + (data["FLAG_WORK_PHONE"])) / 2.0))),
                                                                                              ((((np.maximum(((data["NEW_SCORES_STD"])), ((np.maximum(((data["ORGANIZATION_TYPE_Transport__type_3"])), ((data["DAYS_ID_PUBLISH"]))))))) - (0.636620))))))))
    v["i21"] = 0.099700 * np.tanh(np.where(data["CC_AMT_DRAWINGS_CURRENT_VAR"] < -99998, 0.318310, (((((((((-1.0 * ((data["CC_NAME_CONTRACT_STATUS_Active_VAR"])))) + (
        data["CC_AMT_DRAWINGS_CURRENT_VAR"]))) + (data["CC_AMT_DRAWINGS_CURRENT_VAR"]))) - (data["CC_NAME_CONTRACT_STATUS_Demand_VAR"]))) * ((((11.63801860809326172)) * 2.0)))))
    v["i22"] = 0.099600 * np.tanh(((((np.where(data["ORGANIZATION_TYPE_Military"] > 0, -2.0, ((data["REG_CITY_NOT_LIVE_CITY"]) - (((((np.maximum(((data["NAME_INCOME_TYPE_State_servant"])),
                                                                                                                                                 ((data["APPROVED_RATE_DOWN_PAYMENT_MAX"])))) * 2.0)) * 2.0))))) + (data["NAME_EDUCATION_TYPE_Lower_secondary"]))) + (data["NAME_EDUCATION_TYPE_Lower_secondary"])))
    v["i23"] = 0.099700 * np.tanh(((((-1.0 * ((data["OCCUPATION_TYPE_Accountants"])))) + ((((data["NAME_HOUSING_TYPE_Municipal_apartment"]) + (np.minimum(
        ((((((((data["NAME_FAMILY_STATUS_Married"]) - (data["OCCUPATION_TYPE_Sales_staff"]))) * (data["ACTIVE_DAYS_CREDIT_VAR"]))) / 2.0))), (((-1.0 * ((data["ORGANIZATION_TYPE_Industry__type_9"])))))))) / 2.0))) / 2.0))
    v["i24"] = 0.097890 * np.tanh(np.where(((data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]) - (data["NAME_INCOME_TYPE_Unemployed"])) < 0, ((np.minimum(
        ((data["NEW_CREDIT_TO_INCOME_RATIO"])), ((((data["ORGANIZATION_TYPE_Transport__type_3"]) * 2.0))))) - (np.tanh((((data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]) - (-1.0)))))), (6.0)))
    v["i25"] = 0.100000 * np.tanh(((data["ORGANIZATION_TYPE_Transport__type_3"]) + (
        ((((((((((((((((data["INSTAL_DPD_MEAN"]) - (data["ORGANIZATION_TYPE_Military"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0))))
    v["i26"] = 0.099698 * np.tanh((((np.where((((data["OCCUPATION_TYPE_Low_skill_Laborers"]) < (data["ACTIVE_AMT_ANNUITY_MAX"])) * 1.) > 0, (8.0), ((((data["OCCUPATION_TYPE_Laborers"]) - (
        data["WALLSMATERIAL_MODE_Panel"]))) - (data["FLAG_PHONE"])))) + (np.maximum(((data["OCCUPATION_TYPE_Low_skill_Laborers"])), ((data["OCCUPATION_TYPE_Cooking_staff"]))))) / 2.0))
    v["i27"] = 0.099970 * np.tanh(((data["ORGANIZATION_TYPE_Trade__type_3"]) - (np.maximum((((((((((((((data["ORGANIZATION_TYPE_Trade__type_3"]) + (data["CC_NAME_CONTRACT_STATUS_Completed_VAR"])) / 2.0)) > (
        np.tanh((np.tanh((data["CC_AMT_DRAWINGS_CURRENT_VAR"])))))) * 1.)) + (data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"])) / 2.0)) * 2.0))), ((data["OCCUPATION_TYPE_High_skill_tech_staff"]))))))
    v["i28"] = 0.099996 * np.tanh(((((np.minimum(((data["NEW_CREDIT_TO_INCOME_RATIO"])), ((((-1.0) / 2.0))))) + (((data["APPROVED_RATE_DOWN_PAYMENT_MEAN"]) - (
        np.where(data["ORGANIZATION_TYPE_Trade__type_2"] < 0, ((((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]) * 2.0)) * 2.0), data["NEW_CREDIT_TO_INCOME_RATIO"])))))) * 2.0))
    v["i29"] = 0.100000 * np.tanh((((13.18957233428955078)) * (((((data["INSTAL_DAYS_ENTRY_PAYMENT_SUM"]) + (((((np.maximum(((3.0)), ((
        (((14.53842449188232422)) * (data["INSTAL_DPD_MEAN"])))))) * (data["INSTAL_DPD_MEAN"]))) * 2.0)))) * ((5.49084424972534180))))))
    v["i30"] = 0.097617 * np.tanh(((np.maximum((((((((data["CC_CNT_DRAWINGS_OTHER_CURRENT_MEAN"]) * 2.0)) > (np.maximum(((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"])), ((data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"]))))) * 1.))),
                                               ((((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) + (data["CC_CNT_DRAWINGS_POS_CURRENT_MEAN"]))) + (((data["CC_CNT_DRAWINGS_POS_CURRENT_MEAN"]) * 2.0))))))) * 2.0))
    v["i31"] = 0.099942 * np.tanh((((5.04678153991699219)) * (np.where(((data["EXT_SOURCE_3"]) + (data["ORGANIZATION_TYPE_Realtor"]))
                                                                       < -99998, data["OCCUPATION_TYPE_Cleaning_staff"], (((-1.0 * ((np.tanh((np.tanh((0.318310)))))))) - (data["EXT_SOURCE_3"]))))))
    v["i32"] = 0.099999 * np.tanh(((((((np.minimum(((data["NEW_CREDIT_TO_INCOME_RATIO"])), (((-1.0 * ((data["HOUR_APPR_PROCESS_START"]))))))) - (
        ((((data["NAME_EDUCATION_TYPE_Incomplete_higher"]) * 2.0)) * 2.0)))) - (data["ORGANIZATION_TYPE_Security_Ministries"]))) - (np.tanh((data["OCCUPATION_TYPE_Core_staff"])))))
    v["i33"] = 0.099756 * np.tanh(np.where(data["YEARS_BEGINEXPLUATATION_MODE"] < -99998, ((np.minimum(((data["AMT_INCOME_TOTAL"])), (((((-1.0 * ((data["ORGANIZATION_TYPE_Industry__type_9"]))))
                                                                                                                                        * (data["ORGANIZATION_TYPE_Industry__type_9"])))))) * 2.0), (((data["NAME_INCOME_TYPE_Unemployed"]) > (data["YEARS_BEGINEXPLUATATION_MODE"])) * 1.)))
    v["i34"] = 0.099040 * np.tanh(((((((data["ORGANIZATION_TYPE_Realtor"]) + (data["FLAG_WORK_PHONE"])) / 2.0)) + (((np.maximum(((data["NAME_INCOME_TYPE_Unemployed"])), ((data["OCCUPATION_TYPE_Security_staff"])))
                                                                                                                     ) + (((data["ORGANIZATION_TYPE_Transport__type_3"]) - ((((((data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"]) / 2.0)) > (data["ORGANIZATION_TYPE_Realtor"])) * 1.))))))) / 2.0))
    v["i35"] = 0.099800 * np.tanh((((((data["ORGANIZATION_TYPE_Realtor"]) + (((((((data["ORGANIZATION_TYPE_Restaurant"]) + (data["NAME_INCOME_TYPE_Maternity_leave"])) / 2.0)) + (data["NAME_INCOME_TYPE_Unemployed"])) / 2.0))) / 2.0)) - (
        np.where(data["ORGANIZATION_TYPE_Restaurant"] > 0, data["WEEKDAY_APPR_PROCESS_START_SUNDAY"], (((data["ORGANIZATION_TYPE_Industry__type_9"]) + (data["WEEKDAY_APPR_PROCESS_START_SUNDAY"])) / 2.0)))))
    v["i36"] = 0.099500 * np.tanh((((((data["ORGANIZATION_TYPE_Self_employed"]) / 2.0)) + ((((-1.0 * ((np.maximum(((np.where(data["WEEKDAY_APPR_PROCESS_START_SATURDAY"] > 0, data["OCCUPATION_TYPE_Accountants"], data[
                                  "OCCUPATION_TYPE_Medicine_staff"]))), (((((data["WEEKDAY_APPR_PROCESS_START_SATURDAY"]) + (data["ORGANIZATION_TYPE_Military"])) / 2.0)))))))) - (data["OCCUPATION_TYPE_Accountants"])))) / 2.0))
    v["i37"] = 0.099450 * np.tanh(np.where((((data["CLOSED_DAYS_CREDIT_MEAN"]) > (1.570796)) * 1.) > 0, 3.0, np.where(data["AMT_INCOME_TOTAL"] > 0, data[
                                  "CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"], np.where(data["WEEKDAY_APPR_PROCESS_START_MONDAY"] > 0, -1.0, (((data["CLOSED_DAYS_CREDIT_MEAN"]) > (1.570796)) * 1.)))))
    v["i38"] = 0.098500 * np.tanh(np.where(((data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"]) + (np.tanh((data["ACTIVE_MONTHS_BALANCE_MIN_MIN"])))) > 0,
                                           (-1.0 * (((10.21496009826660156)))), np.maximum(((0.318310)), ((((data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"]) + ((10.42618179321289062))))))))
    v["i39"] = 0.097940 * np.tanh((-1.0 * ((((data["ORGANIZATION_TYPE_Police"]) - (((data["ORGANIZATION_TYPE_Realtor"]) + (((data["ORGANIZATION_TYPE_Construction"]) + (
        (((data["NAME_INCOME_TYPE_Maternity_leave"]) + (((data["ORGANIZATION_TYPE_Construction"]) + (data["NAME_INCOME_TYPE_Unemployed"])))) / 2.0)))))))))))
    v["i40"] = 0.099370 * np.tanh(((np.where(data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"] > 0, ((((((0.318310) - (data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"]))) * 2.0)) * 2.0),
                                             (((((data["NAME_INCOME_TYPE_Unemployed"]) + (data["NAME_EDUCATION_TYPE_Lower_secondary"])) / 2.0)) - (data["ORGANIZATION_TYPE_Bank"])))) * 2.0))
    v["i41"] = 0.099963 * np.tanh(((((((((((((((data["NEW_CREDIT_TO_INCOME_RATIO"]) * ((-1.0 * ((data["AMT_INCOME_TOTAL"])))))) * 2.0)) * 2.0)) - (
        np.maximum(((data["AMT_INCOME_TOTAL"])), ((data["NEW_CREDIT_TO_INCOME_RATIO"])))))) - (data["NAME_HOUSING_TYPE_Office_apartment"]))) * 2.0)) * 2.0))
    v["i42"] = 0.099999 * np.tanh((((((-1.0 * ((((data["BURO_STATUS_0_MEAN_MEAN"]) + (np.where(data["BURO_STATUS_0_MEAN_MEAN"] > 0, data["BURO_MONTHS_BALANCE_SIZE_MEAN"],
                                                                                               (((data["BURO_MONTHS_BALANCE_SIZE_MEAN"]) > (-1.0)) * 1.)))))))) * ((((data["BURO_STATUS_0_MEAN_MEAN"]) > (-1.0)) * 1.)))) * 2.0))
    v["i43"] = 0.093012 * np.tanh((((((-1.0 * ((((((((((-1.0 * ((data["ORGANIZATION_TYPE_Trade__type_2"])))) - (np.maximum(((data["NEW_CAR_TO_EMPLOY_RATIO"])), ((
        data["ORGANIZATION_TYPE_Industry__type_12"])))))) < (data["REG_REGION_NOT_LIVE_REGION"])) * 1.)) * 2.0))))) - (data["REG_REGION_NOT_LIVE_REGION"]))) * 2.0))
    v["i44"] = 0.082010 * np.tanh(np.where(data["AMT_INCOME_TOTAL"] < 0, ((((((((1.570796) * (np.where(data["NAME_HOUSING_TYPE_Rented_apartment"] > 0, data["NAME_HOUSING_TYPE_Rented_apartment"], data[
                                  "AMT_INCOME_TOTAL"])))) - (data["NAME_FAMILY_STATUS_Widow"]))) * 2.0)) - (data["NAME_FAMILY_STATUS_Widow"])), data["NAME_HOUSING_TYPE_Rented_apartment"]))
    v["i45"] = 0.098100 * np.tanh(((((((data["NAME_INCOME_TYPE_Maternity_leave"]) + (data["NAME_INCOME_TYPE_Unemployed"])) / 2.0)) + (((((data["OCCUPATION_TYPE_Cleaning_staff"]) - (
        data["ORGANIZATION_TYPE_Industry__type_5"]))) - ((((data["ORGANIZATION_TYPE_School"]) + (data["NAME_EDUCATION_TYPE_Academic_degree"])) / 2.0))))) / 2.0))
    v["i46"] = 0.099505 * np.tanh((((((-1.0 * (((((data["NAME_INCOME_TYPE_Maternity_leave"]) > ((((data["NEW_CREDIT_TO_INCOME_RATIO"]) + (
        (((((data["AMT_INCOME_TOTAL"]) / 2.0)) + (2.0)) / 2.0))) / 2.0))) * 1.))))) - (data["ORGANIZATION_TYPE_Hotel"]))) - (((data["NAME_EDUCATION_TYPE_Academic_degree"]) / 2.0))))
    v["i47"] = 0.099995 * np.tanh(((((data["FLAG_OWN_REALTY"]) * (data["ORGANIZATION_TYPE_Trade__type_7"]))) + (np.maximum((((((data["NAME_INCOME_TYPE_Unemployed"]) + (data["NAME_INCOME_TYPE_Maternity_leave"])) / 2.0))),
                                                                                                                           ((np.maximum(((((data["ORGANIZATION_TYPE_Trade__type_3"]) + (data["ORGANIZATION_TYPE_Legal_Services"])))), ((data["ORGANIZATION_TYPE_Trade__type_7"])))))))))
    v["i48"] = 0.098997 * np.tanh((((data["NAME_TYPE_SUITE_Children"]) + (((((((data["NAME_INCOME_TYPE_Unemployed"]) + (data["NAME_FAMILY_STATUS_Separated"])) / 2.0)) + (np.where(data["NAME_FAMILY_STATUS_Separated"]
                                                                                                                                                                                   < 0, np.maximum(((data["NAME_HOUSING_TYPE_Municipal_apartment"])), ((data["NAME_TYPE_SUITE_Other_B"]))), ((data["AMT_INCOME_TOTAL"]) - (data["NAME_TYPE_SUITE_Other_B"]))))) / 2.0))) / 2.0))
    v["i49"] = 0.071280 * np.tanh((((((((-1.0 * ((data["CODE_GENDER"])))) + (data["ORGANIZATION_TYPE_Transport__type_4"])) / 2.0)) + (((((((data["NEW_CREDIT_TO_INCOME_RATIO"]) + (
        (((data["CODE_GENDER"]) + (data["ORGANIZATION_TYPE_Realtor"])) / 2.0))) / 2.0)) + (((data["OCCUPATION_TYPE_Low_skill_Laborers"]) - (data["OCCUPATION_TYPE_Private_service_staff"])))) / 2.0))) / 2.0))
    v["i50"] = 0.090101 * np.tanh((((data["OCCUPATION_TYPE_Waiters_barmen_staff"]) + (((data["NAME_INCOME_TYPE_Maternity_leave"]) - (((((((0.318310) < (np.where(data["NEW_INC_PER_CHLD"] > 0, data[
                                  "AMT_INCOME_TOTAL"], data["ORGANIZATION_TYPE_Security_Ministries"]))) * 1.)) + ((((data["NEW_INC_BY_ORG"]) + (data["ORGANIZATION_TYPE_Security_Ministries"])) / 2.0))) / 2.0))))) / 2.0))
    v["i51"] = 0.077600 * np.tanh(np.where((((data["AMT_INCOME_TOTAL"]) + (data["HOUSETYPE_MODE_specific_housing"])) / 2.0) > 0, np.maximum(((np.minimum(((data["ORGANIZATION_TYPE_Business_Entity_Type_3"])), ((
        data["HOUSETYPE_MODE_specific_housing"]))))), ((((data["ORGANIZATION_TYPE_Business_Entity_Type_3"]) * (np.maximum(((data["HOUSETYPE_MODE_specific_housing"])), ((data["AMT_INCOME_TOTAL"])))))))), data["ORGANIZATION_TYPE_Legal_Services"]))
    v["i52"] = 0.084003 * np.tanh((((((((((data["ORGANIZATION_TYPE_Legal_Services"]) > (((data["NEW_INC_PER_CHLD"]) + (np.tanh((1.0)))))) * 1.)) + (
        data["ORGANIZATION_TYPE_Realtor"]))) - (data["ORGANIZATION_TYPE_Industry__type_12"]))) - (data["NAME_EDUCATION_TYPE_Academic_degree"])))
    v["i53"] = 0.096301 * np.tanh((((((((((((data["AMT_INCOME_TOTAL"]) + (data["REFUSED_AMT_DOWN_PAYMENT_MAX"]))) + (data["REFUSED_AMT_DOWN_PAYMENT_MAX"]))) * (
        np.tanh((0.318310))))) > (np.tanh((data["REFUSED_AMT_DOWN_PAYMENT_MEAN"])))) * 1.)) + (data["HOUSETYPE_MODE_specific_housing"])))
    v["i54"] = 0.098989 * np.tanh(np.minimum((((((3.0)) - (data["REFUSED_AMT_ANNUITY_MAX"])))), ((np.minimum((((((0.0) < (data["REFUSED_APP_CREDIT_PERC_MAX"])) * 1.))),
                                                                                                             ((((data["REFUSED_AMT_ANNUITY_MAX"]) - (((data["REFUSED_APP_CREDIT_PERC_MAX"]) + (data["REFUSED_AMT_GOODS_PRICE_MEAN"])))))))))))
    v["i55"] = 0.094598 * np.tanh(np.maximum(((data["ORGANIZATION_TYPE_Transport__type_3"])), ((np.where((((((data["AMT_INCOME_TOTAL"]) < (data["ORGANIZATION_TYPE_Trade__type_6"])) * 1.)) * (
        data["NAME_INCOME_TYPE_Maternity_leave"])) < 0, data["NAME_INCOME_TYPE_State_servant"], (-1.0 * ((np.maximum(((data["NAME_INCOME_TYPE_State_servant"])), ((data["ORGANIZATION_TYPE_Transport__type_3"])))))))))))
    v["i56"] = 0.089650 * np.tanh(((((np.minimum(((((data["OCCUPATION_TYPE_Managers"]) * (((np.tanh((((data["NAME_TYPE_SUITE_Spouse__partner"]) + (np.tanh(
        (data["AMT_INCOME_TOTAL"]))))))) + (data["NAME_EDUCATION_TYPE_Academic_degree"])))))), (((-1.0 * ((data["ORGANIZATION_TYPE_Trade__type_6"]))))))) * 2.0)) * 2.0))
    v["i57"] = 0.100000 * np.tanh(np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"] < -99998, data["ORGANIZATION_TYPE_Legal_Services"], np.where(data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"] > 0, ((
        ((((data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"]) + (data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"]))) - (data["AMT_INCOME_TOTAL"]))) + (data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"])), data["NEW_INC_PER_CHLD"])))
    v["i58"] = 0.099530 * np.tanh((((((((((((data["AMT_INCOME_TOTAL"]) + (data["NAME_INCOME_TYPE_Unemployed"])) / 2.0)) + (((data["ORGANIZATION_TYPE_Restaurant"]) - (
        data["ORGANIZATION_TYPE_Electricity"])))) / 2.0)) - (data["ORGANIZATION_TYPE_Industry__type_5"]))) + ((((data["AMT_INCOME_TOTAL"]) > (((2.0) / 2.0))) * 1.))) / 2.0))
    v["i59"] = 0.099529 * np.tanh(((((((((data["OCCUPATION_TYPE_IT_staff"]) * (data["ORGANIZATION_TYPE_Industry__type_12"]))) - ((((((np.minimum(((data["OCCUPATION_TYPE_IT_staff"])), ((
        data["NEW_INC_PER_CHLD"])))) / 2.0)) > (((data["FONDKAPREMONT_MODE_org_spec_account"]) * (data["NAME_EDUCATION_TYPE_Academic_degree"])))) * 1.)))) * 2.0)) * 2.0))
    v["i60"] = 0.099740 * np.tanh(((((data["DEF_60_CNT_SOCIAL_CIRCLE"]) + (((np.minimum(((data["ORGANIZATION_TYPE_Police"])), ((np.minimum(((data["FLAG_EMP_PHONE"])), ((data["NEW_INC_PER_CHLD"]))))))) * (
        np.where(data["ORGANIZATION_TYPE_Transport__type_1"] > 0, data["ORGANIZATION_TYPE_Transport__type_1"], data["ORGANIZATION_TYPE_Police"])))))) + (data["NAME_INCOME_TYPE_Unemployed"])))
    v["i61"] = 0.099555 * np.tanh((-1.0 * ((np.where(data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"] < 0, np.where(data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"] < -99998, (-1.0 * ((np.maximum(
        ((data["DAYS_REGISTRATION"])), ((0.0)))))), data["ORGANIZATION_TYPE_Police"]), ((data["DAYS_REGISTRATION"]) + (((data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"]) / 2.0))))))))
    v["i62"] = 0.100000 * np.tanh(((((((-1.0) / 2.0)) - (np.where(data["EXT_SOURCE_3"] < -99998, 0.0, ((data["EXT_SOURCE_3"]) * 2.0))))) - (
        (((np.tanh((((data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]) / 2.0)))) < (data["EXT_SOURCE_3"])) * 1.))))
    v["i63"] = 0.098850 * np.tanh(np.where(((((((data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]) > ((-1.0 * ((0.318310))))) * 1.)) < (((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) - (
        data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"])))) * 1.) > 0, -1.0, ((((-1.0 * ((0.318310)))) > (data["REFUSED_AMT_ANNUITY_MEAN"])) * 1.)))
    v["i64"] = 0.099952 * np.tanh((-1.0 * ((((np.where(data["EXT_SOURCE_3"] < -99998, np.where(data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"] < 0, 0.318310, data[
                                  "EXT_SOURCE_3"]), ((data["EXT_SOURCE_3"]) - (np.tanh((((np.tanh((data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]))) * 2.0))))))) * 2.0)))))
    v["i65"] = 0.099900 * np.tanh(np.where(data["NEW_SOURCES_PROD"] < -99998, 0.318310, (-1.0 * ((((((data["NEW_SOURCES_PROD"]) - (
        (-1.0 * (((((((data["NEW_SOURCES_PROD"]) > (-1.0)) * 1.)) - (data["ORGANIZATION_TYPE_XNA"])))))))) * ((8.0))))))))
    v["i66"] = 0.095998 * np.tanh(np.minimum(((np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, 0.636620, ((data["ACTIVE_DAYS_CREDIT_VAR"]) / 2.0)))), ((np.where(
        ((data["NEW_CREDIT_TO_INCOME_RATIO"]) + (data["ACTIVE_DAYS_CREDIT_VAR"])) < -99998, (-1.0 * ((data["REFUSED_DAYS_DECISION_MEAN"]))), data["NEW_CREDIT_TO_INCOME_RATIO"])))))
    v["i67"] = 0.099733 * np.tanh((((((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) < (data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"])) * 1.)) - (np.where(data["EXT_SOURCE_3"] < -99998, ((
        ((1.0)) > (data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"])) * 1.), data["EXT_SOURCE_3"])))) - ((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) > (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])) * 1.))))
    v["i68"] = 0.098499 * np.tanh(np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, 0.318310, ((((((np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])), ((data["ACTIVE_AMT_CREDIT_SUM_SUM"]))))
                                                                                                   * 2.0)) - (((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) * (((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) * (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]))))))) * 2.0)))
    v["i69"] = 0.099999 * np.tanh(np.where((((((data["INSTAL_AMT_INSTALMENT_MAX"]) * 2.0)) + ((((3.0) > (data["NEW_SCORES_STD"])) * 1.))) / 2.0) < 0, np.where(
        data["INSTAL_DAYS_ENTRY_PAYMENT_SUM"] > 0, (8.0), data["INSTAL_DAYS_ENTRY_PAYMENT_SUM"]), ((((data["INSTAL_AMT_INSTALMENT_MAX"]) * 2.0)) * 2.0)))
    v["i70"] = 0.099998 * np.tanh(((((data["NEW_SCORES_STD"]) / 2.0)) - (np.minimum((((((np.tanh((data["ACTIVE_MONTHS_BALANCE_MAX_MAX"]))) + (
        3.141593)) / 2.0))), ((np.maximum(((((((((data["NEW_SCORES_STD"]) * 2.0)) * 2.0)) * 2.0))), ((data["NEW_SCORES_STD"])))))))))
    v["i71"] = 0.099949 * np.tanh(np.where((((data["NEW_SOURCES_PROD"]) + (np.tanh((1.0)))) / 2.0) < 0, np.where(
        data["NEW_SOURCES_PROD"] < -99998, data["NAME_INCOME_TYPE_Unemployed"], (12.80127429962158203)), (-1.0 * (((12.80127429962158203))))))
    v["i72"] = 0.099500 * np.tanh((((((data["NEW_SOURCES_PROD"]) > (((-2.0) * 2.0))) * 1.)) * (
        (((((((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) < (data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"])) * 1.)) - (0.636620))) - (data["NEW_SOURCES_PROD"]))) * 2.0))))
    v["i73"] = 0.099052 * np.tanh(np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, (((0.318310) > (((data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]) + (
        ((((((-1.0 * ((data["REFUSED_CNT_PAYMENT_MEAN"])))) > (0.318310)) * 1.)) / 2.0))))) * 1.), (((data["ACTIVE_DAYS_CREDIT_VAR"]) + (data["ORGANIZATION_TYPE_XNA"])) / 2.0)))
    v["i74"] = 0.099260 * np.tanh((((((-1.0 * (((((data["EXT_SOURCE_1"]) > (((np.minimum(
        (((((data["NEW_SOURCES_PROD"]) < (((0.636620) * 2.0))) * 1.))), ((0.636620)))) + (np.tanh((0.318310)))))) * 1.))))) * 2.0)) * 2.0))
    v["i75"] = 0.099504 * np.tanh((-1.0 * ((np.where(np.where(data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"] < 0, data["EXT_SOURCE_1"], 0.318310) < -99998, 0.318310, np.where(
        data["NEW_SOURCES_PROD"] < -99998, data["EXT_SOURCE_1"], np.where(data["NEW_SOURCES_PROD"] < 0, data["NEW_SOURCES_PROD"], data["EXT_SOURCE_1"])))))))
    v["i76"] = 0.099849 * np.tanh(np.where((((0.636620) < (((data["DAYS_BIRTH"]) / 2.0))) * 1.) > 0, -2.0, (((((data["DAYS_BIRTH"]) * (
        data["DAYS_BIRTH"]))) < (np.where(data["DAYS_BIRTH"] > 0, 0.636620, data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]))) * 1.)))
    v["i77"] = 0.099975 * np.tanh((((-1.0 * ((np.where(data["REFUSED_CNT_PAYMENT_MEAN"] > 0, data["REFUSED_AMT_CREDIT_MEAN"], (((((np.minimum(((np.maximum(
        ((np.tanh((data["REFUSED_AMT_ANNUITY_MAX"])))), ((data["REFUSED_AMT_ANNUITY_MAX"]))))), ((data["REFUSED_AMT_APPLICATION_MAX"])))) < (data["REFUSED_AMT_ANNUITY_MEAN"])) * 1.)) * 2.0)))))) * 2.0))
    v["i78"] = 0.099301 * np.tanh(np.minimum(((((1.0) - (data["DAYS_BIRTH"])))), ((np.where(data["OCCUPATION_TYPE_Laborers"] > 0, np.maximum(
        ((data["ACTIVE_MONTHS_BALANCE_MIN_MIN"])), ((data["DAYS_BIRTH"]))), (((((data["NEW_SCORES_STD"]) / 2.0)) < (data["NEW_INC_PER_CHLD"])) * 1.))))))
    v["i79"] = 0.097599 * np.tanh(np.minimum(((np.where(data["ACTIVE_MONTHS_BALANCE_MIN_MIN"] < 0, ((((((data["NAME_CONTRACT_TYPE_Revolving_loans"]) - (
        data["NAME_FAMILY_STATUS_Married"]))) / 2.0)) / 2.0), data["NAME_FAMILY_STATUS_Married"]))), ((((data["NAME_FAMILY_STATUS_Married"]) - (data["ACTIVE_MONTHS_BALANCE_MIN_MIN"]))))))
    v["i80"] = 0.094203 * np.tanh(np.where(data["AMT_INCOME_TOTAL"] < 0, ((data["NEW_DOC_IND_KURT"]) * (data["AMT_INCOME_TOTAL"])), ((data["NEW_DOC_IND_KURT"]) + ((((np.minimum(
        ((data["FONDKAPREMONT_MODE_reg_oper_spec_account"])), ((data["OCCUPATION_TYPE_Core_staff"])))) + (((data["OCCUPATION_TYPE_Core_staff"]) - (data["FONDKAPREMONT_MODE_reg_oper_spec_account"])))) / 2.0)))))
    v["i81"] = 0.099984 * np.tanh(np.where(np.minimum(((data["DAYS_ID_PUBLISH"])), ((np.minimum(((data["NEW_DOC_IND_KURT"])), ((data["NEW_CREDIT_TO_INCOME_RATIO"])))))) < 0, np.where(
        data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"] < -99998, data["DAYS_ID_PUBLISH"], ((data["NEW_CREDIT_TO_INCOME_RATIO"]) + (data["NEW_DOC_IND_KURT"]))), data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]))
    v["i82"] = 0.099600 * np.tanh((((2.0) < ((((((((((data["ORGANIZATION_TYPE_Business_Entity_Type_3"]) * (data["FLAG_CONT_MOBILE"]))) * 2.0)) + (
        3.0))) + (((((0.33989319205284119)) + (np.maximum(((data["DAYS_ID_PUBLISH"])), ((data["AMT_INCOME_TOTAL"]))))) / 2.0))) / 2.0))) * 1.))
    v["i83"] = 0.073300 * np.tanh(((((((data["WALLSMATERIAL_MODE_Others"]) + ((((data["ORGANIZATION_TYPE_Realtor"]) + (0.636620)) / 2.0))) / 2.0)) + (
        (((-1.0 * ((data["NAME_EDUCATION_TYPE_Academic_degree"])))) + (((data["ORGANIZATION_TYPE_Mobile"]) - ((((0.318310) < (data["AMT_INCOME_TOTAL"])) * 1.))))))) / 2.0))
    v["i84"] = 0.081498 * np.tanh(((((((data["NAME_EDUCATION_TYPE_Higher_education"]) + (((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]) * (data["AMT_INCOME_TOTAL"])))) / 2.0)) + (
        ((data["AMT_INCOME_TOTAL"]) * (((data["AMT_INCOME_TOTAL"]) * (((data["AMT_INCOME_TOTAL"]) * ((-1.0 * ((data["NAME_EDUCATION_TYPE_Higher_education"]))))))))))) / 2.0))
    v["i85"] = 0.099970 * np.tanh(((data["ORGANIZATION_TYPE_Legal_Services"]) + (np.where((((2.0) < (((data["ORGANIZATION_TYPE_Self_employed"]) + (data["NEW_INC_PER_CHLD"]))))
                                                                                           * 1.) > 0, (((data["NEW_INC_PER_CHLD"]) < (((data["ORGANIZATION_TYPE_Legal_Services"]) * 2.0))) * 1.), data["NAME_INCOME_TYPE_Maternity_leave"]))))
    v["i86"] = 0.098020 * np.tanh(np.where(data["REGION_RATING_CLIENT_W_CITY"] > 0, np.minimum(((((data["ORGANIZATION_TYPE_Agriculture"]) - (data["NEW_INC_PER_CHLD"])))), ((
        (((data["ORGANIZATION_TYPE_Agriculture"]) + (data["ORGANIZATION_TYPE_Business_Entity_Type_3"])) / 2.0)))), (((((data["NEW_INC_PER_CHLD"]) + (data["REGION_RATING_CLIENT"]))) > (1.570796)) * 1.)))
    v["i87"] = 0.094996 * np.tanh(((((data["NAME_INCOME_TYPE_Maternity_leave"]) - (np.where(data["AMT_INCOME_TOTAL"] < 0, np.where(data["NEW_INC_PER_CHLD"] < 0, ((data["ORGANIZATION_TYPE_Industry__type_2"]) / 2.0), data[
                                  "REG_REGION_NOT_LIVE_REGION"]), np.maximum(((data["ORGANIZATION_TYPE_Industry__type_2"])), ((data["REG_REGION_NOT_LIVE_REGION"]))))))) - (data["NAME_EDUCATION_TYPE_Academic_degree"])))
    v["i88"] = 0.099811 * np.tanh(np.where(((((data["WEEKDAY_APPR_PROCESS_START_TUESDAY"]) * (data["WEEKDAY_APPR_PROCESS_START_TUESDAY"]))) - (
        data["AMT_CREDIT"])) > 0, np.maximum(((data["AMT_INCOME_TOTAL"])), ((0.0))), ((data["WEEKDAY_APPR_PROCESS_START_TUESDAY"]) - (data["AMT_INCOME_TOTAL"]))))
    v["i89"] = 0.099800 * np.tanh(((((((((((data["ORGANIZATION_TYPE_Transport__type_4"]) + ((((((data["NAME_EDUCATION_TYPE_Academic_degree"]) * (data["ORGANIZATION_TYPE_Transport__type_4"]))) + (
        data["NEW_CREDIT_TO_INCOME_RATIO"])) / 2.0))) / 2.0)) > ((((data["NEW_CREDIT_TO_INCOME_RATIO"]) > (0.318310)) * 1.))) * 1.)) - (data["NAME_EDUCATION_TYPE_Academic_degree"]))) * 2.0))
    v["i90"] = 0.098490 * np.tanh(np.maximum(((data["ORGANIZATION_TYPE_Industry__type_1"])), ((((data["NEW_CREDIT_TO_INCOME_RATIO"]) - (np.where(
        (((np.where(data["NEW_CREDIT_TO_INCOME_RATIO"] < 0, data["AMT_INCOME_TOTAL"], data["NEW_INC_PER_CHLD"])) + (data["NAME_INCOME_TYPE_Unemployed"])) / 2.0) > 0, data["NEW_INC_PER_CHLD"], (6.0))))))))
    v["i91"] = 0.093999 * np.tanh(((data["NAME_INCOME_TYPE_Unemployed"]) + (((((((((data["NAME_INCOME_TYPE_Maternity_leave"]) + ((((((data["WEEKDAY_APPR_PROCESS_START_WEDNESDAY"]) + (
        data["ORGANIZATION_TYPE_Business_Entity_Type_3"])) / 2.0)) - (data["NAME_INCOME_TYPE_Student"])))) / 2.0)) / 2.0)) + (((data["NAME_INCOME_TYPE_Unemployed"]) - (data["ORGANIZATION_TYPE_Transport__type_1"])))) / 2.0))))
    v["i92"] = 0.098000 * np.tanh(np.minimum(((np.where(data["EXT_SOURCE_3"] < -99998, data["NAME_INCOME_TYPE_State_servant"], np.where(
        (((1.0) + (data["EXT_SOURCE_3"])) / 2.0) > 0, data["EXT_SOURCE_3"], (8.0))))), ((((0.636620) - (data["EXT_SOURCE_3"]))))))
    v["i93"] = 0.099730 * np.tanh((((-1.0 * ((((((np.where((((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]) > ((((data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]) > (data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"])) * 1.)))
                                                            * 1.) > 0, data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"], (((data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]) > (data["ACTIVE_CNT_CREDIT_PROLONG_SUM"])) * 1.))) * 2.0)) * 2.0))))) * 2.0))
    v["i94"] = 0.098545 * np.tanh(np.minimum(((np.where((((1.0) < (data["AMT_REQ_CREDIT_BUREAU_QRT"])) * 1.) > 0, data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"], ((
        data["AMT_REQ_CREDIT_BUREAU_QRT"]) * (-2.0))))), (((((data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]) > (np.tanh((data["AMT_REQ_CREDIT_BUREAU_QRT"])))) * 1.)))))
    v["i95"] = 0.095403 * np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"] < -99998, 0.318310, ((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) * (np.where(np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]
                                                                                                                                                                         > 0, data["ACTIVE_CNT_CREDIT_PROLONG_SUM"], (-1.0 * ((data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"])))) > 0, 3.141593, (-1.0 * ((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]))))))))
    v["i96"] = 0.098999 * np.tanh(((((np.maximum(((((((data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]) - (data["CLOSED_AMT_CREDIT_SUM_SUM"]))) * 2.0))), ((np.minimum(
        ((data["CLOSED_DAYS_CREDIT_MEAN"])), ((data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]))))))) * 2.0)) - ((((data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]) > (1.0)) * 1.))))
    v["i97"] = 0.076491 * np.tanh(np.where(np.where(data["NEW_SCORES_STD"] > 0, data["DAYS_BIRTH"], data["ENTRANCES_MODE"]) > 0, ((
        (((data["DAYS_BIRTH"]) < (data["ENTRANCES_MODE"])) * 1.)) - (data["DAYS_BIRTH"])), ((np.tanh((np.tanh((data["DAYS_BIRTH"]))))) / 2.0)))
    v["i98"] = 0.049979 * np.tanh(((np.where((((((data["CLOSED_AMT_ANNUITY_MEAN"]) + (data["NAME_INCOME_TYPE_Student"]))) + (data["ORGANIZATION_TYPE_Hotel"])) / 2.0)
                                             < 0, ((data["NAME_INCOME_TYPE_Working"]) * (data["ORGANIZATION_TYPE_Hotel"])), data["NAME_INCOME_TYPE_Working"])) * 2.0))
    v["i99"] = 0.071179 * np.tanh(np.where(data["EXT_SOURCE_3"] < -99998, (-1.0 * (((((((0.0) > (data["AMT_INCOME_TOTAL"])) * 1.)) / 2.0)))),
                                           ((np.tanh((np.where(data["EXT_SOURCE_3"] > 0, 0.636620, data["EXT_SOURCE_3"])))) - (data["EXT_SOURCE_3"]))))
    v["i100"] = 0.095300 * np.tanh(np.minimum((((((data["AMT_REQ_CREDIT_BUREAU_WEEK"]) < (data["AMT_REQ_CREDIT_BUREAU_YEAR"])) * 1.))), ((((np.maximum(((data["AMT_REQ_CREDIT_BUREAU_DAY"])), ((
        np.maximum((((((data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]) + (data["AMT_REQ_CREDIT_BUREAU_QRT"])) / 2.0))), ((0.318310))))))) - (data["AMT_REQ_CREDIT_BUREAU_YEAR"]))))))
    v["i101"] = 0.097997 * np.tanh(np.tanh((np.where(data["AMT_REQ_CREDIT_BUREAU_QRT"] < 0, np.where((((((data["AMT_REQ_CREDIT_BUREAU_QRT"]) > (
        data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"])) * 1.)) + (data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"])) > 0, (-1.0 * ((0.636620))), 0.318310), data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]))))
    v["i102"] = 0.091041 * np.tanh(np.minimum((((((-1.0 * ((np.maximum(((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])), ((data["DAYS_BIRTH"]))))))) + (
        (((-1.0 * ((data["ORGANIZATION_TYPE_Hotel"])))) + (1.0)))))), (((((((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) / 2.0)) > (data["ORGANIZATION_TYPE_Hotel"])) * 1.)))))
    v["i103"] = 0.098532 * np.tanh(np.where(data["ORGANIZATION_TYPE_Medicine"] > 0, data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"], (((-1.0 * ((((np.minimum(((0.318310)), ((np.where(
        data["NAME_INCOME_TYPE_Maternity_leave"] > 0, data["NAME_INCOME_TYPE_Maternity_leave"], data["NEW_INC_PER_CHLD"]))))) * (data["NAME_FAMILY_STATUS_Single___not_married"])))))) / 2.0)))
    v["i104"] = 0.080961 * np.tanh(np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, np.where(data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"] < 0, (0.16092185676097870), data["ACTIVE_DAYS_CREDIT_VAR"]),
                                            np.where((((data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]) + (0.318310)) / 2.0) > 0, (((data["ORGANIZATION_TYPE_Mobile"]) + (0.318310)) / 2.0), data["ACTIVE_DAYS_CREDIT_VAR"])))
    v["i105"] = 0.095005 * np.tanh(np.where((((data["EXT_SOURCE_3"]) + (-1.0)) / 2.0) < 0, np.maximum(((((data["WALLSMATERIAL_MODE_Stone__brick"]) * (data["AMT_INCOME_TOTAL"])))),
                                                                                                      ((data["NAME_INCOME_TYPE_Maternity_leave"]))), ((((data["CLOSED_AMT_ANNUITY_MEAN"]) + (data["WALLSMATERIAL_MODE_Stone__brick"]))) * 2.0)))
    v["i106"] = 0.096790 * np.tanh(np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"] < 0, ((np.where(data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"] > 0, data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"], data["NAME_INCOME_TYPE_Maternity_leave"]))
                                                                                        * 2.0), np.where(data["EXT_SOURCE_3"] < 0, ((np.maximum(((data["NAME_INCOME_TYPE_Maternity_leave"])), ((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"])))) * 2.0), data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"])))
    v["i107"] = 0.019000 * np.tanh((((9.0)) * ((((data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]) > (np.where(((((data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]) - ((-1.0 * (
        (data["ACTIVE_DAYS_CREDIT_VAR"])))))) - ((-1.0 * ((data["CLOSED_DAYS_CREDIT_MEAN"]))))) < -99998, -2.0, data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]))) * 1.))))
    v["i108"] = 0.098660 * np.tanh((((((data["EXT_SOURCE_3"]) - (((1.570796) - (data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]))))) > (np.where(data["ACTIVE_DAYS_CREDIT_VAR"]
                                                                                                                                         < 0, np.where(data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"] < 0, data["ACTIVE_DAYS_CREDIT_VAR"], 3.0), ((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) * 2.0)))) * 1.))
    v["i109"] = 0.099920 * np.tanh(((((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (
        ((((3.141593) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)))) * 2.0))), (((-1.0 * ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) * 2.0))) * 2.0)) * 2.0))
    v["i110"] = 0.099820 * np.tanh(np.minimum(((((data["NEW_CREDIT_TO_INCOME_RATIO"]) + (data["NAME_CONTRACT_TYPE_Cash_loans"])))), ((np.where(data["DAYS_BIRTH"] > 0, data[
                                   "NEW_DOC_IND_KURT"], np.where(data["AMT_CREDIT"] > 0, ((data["NAME_CONTRACT_TYPE_Cash_loans"]) + (data["NEW_CREDIT_TO_INCOME_RATIO"])), np.tanh((-1.0))))))))
    v["i111"] = 0.099709 * np.tanh(np.where(data["BURO_MONTHS_BALANCE_MAX_MAX"] < 0, (((data["BURO_MONTHS_BALANCE_MIN_MIN"]) > (data["NAME_INCOME_TYPE_Maternity_leave"]))
                                                                                      * 1.), np.tanh(((((((-1.0) - (data["BURO_STATUS_0_MEAN_MEAN"]))) + ((((-1.0) > (data["BURO_STATUS_0_MEAN_MEAN"])) * 1.))) / 2.0)))))
    v["i112"] = 0.099001 * np.tanh(np.where(data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"] > 0, data["AMT_INCOME_TOTAL"], ((((((-1.0 * ((-1.0)))) / 2.0)) < (((np.where(
        data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"] < -99998, data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"], 0.318310)) - (data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"])))) * 1.)))
    v["i113"] = 0.099000 * np.tanh(((np.minimum(((np.where(data["NEW_EMPLOY_TO_BIRTH_RATIO"] < 0, (-1.0 * ((np.where(data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"] < 0, data[
                                   "NEW_EMPLOY_TO_BIRTH_RATIO"], (((data["NEW_EMPLOY_TO_BIRTH_RATIO"]) > (-1.0)) * 1.))))), -1.0))), ((data["DAYS_EMPLOYED"])))) - (data["NEW_EMPLOY_TO_BIRTH_RATIO"])))
    v["i114"] = 0.098000 * np.tanh(np.where(((data["DAYS_BIRTH"]) - (-2.0)) > 0, np.where((((1.42633831501007080)) - (data["DAYS_BIRTH"]))
                                                                                          > 0, ((((-1.0 * (((1.42633831501007080))))) > (data["DAYS_BIRTH"])) * 1.), -2.0), data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]))
    v["i115"] = 0.095020 * np.tanh((((np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"])), (((((np.tanh((data["ACTIVE_DAYS_CREDIT_MEAN"]))) + (data["ACTIVE_DAYS_CREDIT_MEAN"])) / 2.0))))) > (
        (((np.where((-1.0 * ((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]))) > 0, data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"], data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])) > (data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"])) * 1.))) * 1.))
    v["i116"] = 0.099950 * np.tanh(np.where(data["EXT_SOURCE_3"] < 0, np.where(data["NEW_SOURCES_PROD"] < -99998, np.where(((0.636620) + (
        data["EXT_SOURCE_1"])) < 0, data["NAME_INCOME_TYPE_Maternity_leave"], data["EXT_SOURCE_3"]), data["EXT_SOURCE_3"]), ((2.0) + (data["EXT_SOURCE_1"]))))
    v["i117"] = 0.089997 * np.tanh((((np.minimum(((data["AMT_INCOME_TOTAL"])), ((((np.where(data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"] < -99998, np.tanh((data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"])), data[
                                   "NEW_INC_PER_CHLD"])) * (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])))))) + ((((np.tanh((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]))) > (data["AMT_INCOME_TOTAL"])) * 1.))) / 2.0))
    v["i118"] = 0.099801 * np.tanh(((((((((((np.where(data["EXT_SOURCE_3"] < -99998, data["NAME_INCOME_TYPE_Maternity_leave"], ((
        (data["EXT_SOURCE_3"]) < (((((0.318310) / 2.0)) - (2.0)))) * 1.))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0))
    v["i119"] = 0.098299 * np.tanh(np.where(np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"] < -99998, 0.636620, data["REG_CITY_NOT_WORK_CITY"]) < 0, (((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])
                                                                                                                                                         > (0.636620)) * 1.), ((np.minimum(((data["REG_CITY_NOT_WORK_CITY"])), ((((data["ORGANIZATION_TYPE_Trade__type_2"]) * 2.0))))) * (data["REG_CITY_NOT_WORK_CITY"]))))
    v["i120"] = 0.097399 * np.tanh(np.where(data["NEW_INC_PER_CHLD"] > 0, np.where(data["NAME_INCOME_TYPE_State_servant"] < 0, ((0.318310) + (data["APPROVED_RATE_DOWN_PAYMENT_MEAN"])),
                                                                                   (((np.tanh(((((data["APPROVED_RATE_DOWN_PAYMENT_MEAN"]) < (0.318310)) * 1.)))) > (data["NEW_INC_PER_CHLD"])) * 1.)), data["NAME_INCOME_TYPE_Maternity_leave"]))
    v["i121"] = 0.058000 * np.tanh((((((((((data["DAYS_REGISTRATION"]) - (-1.0))) - (data["NEW_INC_PER_CHLD"]))) * (data["ORGANIZATION_TYPE_Agriculture"]))) + (
        np.maximum(((data["ORGANIZATION_TYPE_Realtor"])), (((((-1.0) > (((data["DAYS_REGISTRATION"]) / 2.0))) * 1.)))))) / 2.0))
    v["i122"] = 0.009503 * np.tanh((((((data["ORGANIZATION_TYPE_Mobile"]) + (((((((((data["AMT_INCOME_TOTAL"]) + (data["AMT_INCOME_TOTAL"]))) + (np.tanh((data["FLAG_PHONE"])))) / 2.0)) + (
        np.maximum(((data["NAME_INCOME_TYPE_Maternity_leave"])), ((data["NAME_INCOME_TYPE_Unemployed"]))))) / 2.0))) / 2.0)) - (data["NAME_INCOME_TYPE_Student"])))
    v["i123"] = 0.029700 * np.tanh((((((data["ORGANIZATION_TYPE_Mobile"]) + (((((data["NAME_INCOME_TYPE_Unemployed"]) + (data["ORGANIZATION_TYPE_Mobile"]))) - (
        data["NAME_INCOME_TYPE_Student"]))))) + ((((data["NAME_INCOME_TYPE_Maternity_leave"]) + ((((((data["FLAG_PHONE"]) + (data["NEW_INC_PER_CHLD"])) / 2.0)) / 2.0))) / 2.0))) / 2.0))
    v["i124"] = 0.099896 * np.tanh(np.where(((data["NEW_SOURCES_PROD"]) - (-1.0)) < 0, (((((-1.0) * 2.0)) < (data["NEW_SOURCES_PROD"])) * 1.), ((
        (((((data["NEW_INC_PER_CHLD"]) * 2.0)) < (data["NEW_SOURCES_PROD"])) * 1.)) * (data["CC_AMT_DRAWINGS_OTHER_CURRENT_MAX"]))))
    v["i125"] = 0.098900 * np.tanh(((np.where(data["REFUSED_DAYS_DECISION_MEAN"] < 0, ((((-1.0 * (((((data["ORGANIZATION_TYPE_Other"]) + (data["REFUSED_AMT_APPLICATION_MEAN"])) / 2.0))))) < (
        (((np.maximum(((data["ORGANIZATION_TYPE_Other"])), ((data["REFUSED_AMT_APPLICATION_MEAN"])))) < (data["REFUSED_DAYS_DECISION_MEAN"])) * 1.))) * 1.), data["OCCUPATION_TYPE_Accountants"])) * 2.0))
    v["i126"] = 0.095459 * np.tanh(np.where(((np.tanh(((2.84455132484436035)))) + (data["NEW_ANNUITY_TO_INCOME_RATIO"])) > 0, (((data["NAME_INCOME_TYPE_Unemployed"]) + (
        ((data["NEW_ANNUITY_TO_INCOME_RATIO"]) / 2.0))) / 2.0), np.maximum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (data["NEW_ANNUITY_TO_INCOME_RATIO"])))))))
    v["i127"] = 0.097680 * np.tanh(np.minimum(((np.where(data["AMT_ANNUITY"] > 0, data["AMT_ANNUITY"], np.minimum(((((data["NEW_CREDIT_TO_ANNUITY_RATIO"])
                                                                                                                     * (-1.0)))), ((data["NAME_INCOME_TYPE_Maternity_leave"])))))), ((((((-1.0 * ((data["AMT_ANNUITY"])))) + (1.570796)) / 2.0)))))
    v["i128"] = 0.096500 * np.tanh(((np.where(data["FLAG_EMAIL"] < 0, (-1.0 * ((np.where(data["NAME_INCOME_TYPE_Maternity_leave"] < 0, ((3.0) * (
        (((data["EXT_SOURCE_3"]) > (1.570796)) * 1.))), data["EXT_SOURCE_3"])))), data["EXT_SOURCE_3"])) - (data["NAME_INCOME_TYPE_Student"])))
    v["i129"] = 0.100000 * np.tanh((((((np.maximum(((data["BURO_STATUS_X_MEAN_MEAN"])), ((np.maximum(((data["EXT_SOURCE_3"])), ((
        ((((-1.0 * ((0.318310)))) > (data["EXT_SOURCE_3"])) * 1.)))))))) < (0.318310)) * 1.)) - ((((1.570796) < (data["EXT_SOURCE_3"])) * 1.))))
    v["i130"] = 0.087069 * np.tanh(np.where(data["BURO_STATUS_0_MEAN_MEAN"] > 0, np.where(data["CLOSED_MONTHS_BALANCE_MAX_MAX"] < 0, ((((((2.0)) > (data["BURO_STATUS_0_MEAN_MEAN"])) * 1.)) + (np.where(
        data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"] < 0, data["NAME_INCOME_TYPE_Working"], data["NAME_INCOME_TYPE_Maternity_leave"]))), data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]), data["NAME_INCOME_TYPE_Maternity_leave"]))
    v["i131"] = 0.099102 * np.tanh(np.where(data["ORGANIZATION_TYPE_XNA"] > 0, data["REGION_POPULATION_RELATIVE"], np.minimum(((data["DAYS_EMPLOYED"])), ((
        (((-1.0 * ((data["DAYS_EMPLOYED"])))) * (((((-1.0 * ((data["DAYS_EMPLOYED"])))) + (data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"])) / 2.0))))))))
    v["i132"] = 0.090200 * np.tanh(np.where(np.minimum(((data["DAYS_REGISTRATION"])), ((data["DAYS_BIRTH"]))) > 0, (-1.0 * ((((data["NEW_INC_BY_ORG"]) / 2.0)))), np.where(
        data["DAYS_REGISTRATION"] < 0, ((data["DAYS_BIRTH"]) * (data["NAME_INCOME_TYPE_State_servant"])), ((data["NAME_INCOME_TYPE_State_servant"]) / 2.0))))
    v["i133"] = 0.099230 * np.tanh(((((((np.tanh((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (0.636620))))) + (0.636620)) / 2.0)) > (((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (
        data["ORGANIZATION_TYPE_Business_Entity_Type_2"])) / 2.0)) > ((((data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]) > (data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.))) * 1.))) * 1.))
    v["i134"] = 0.095024 * np.tanh(np.where(((((((data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]) < (np.tanh((data["ORGANIZATION_TYPE_Business_Entity_Type_2"])))) * 1.)) < (data["NEW_ANNUITY_TO_INCOME_RATIO"]))
                                             * 1.) > 0, data["ORGANIZATION_TYPE_Kindergarten"], ((np.tanh((data["ORGANIZATION_TYPE_Business_Entity_Type_2"]))) - (((data["ORGANIZATION_TYPE_Kindergarten"]) / 2.0)))))
    v["i135"] = 0.099960 * np.tanh(((((np.where((-1.0 * ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) > 0, (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (data["NAME_INCOME_TYPE_Working"])) * 1.),
                                                np.tanh((data["NAME_INCOME_TYPE_Working"])))) + (np.minimum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), (((-1.0 * ((data["NAME_INCOME_TYPE_Working"]))))))))) * 2.0))
    v["i136"] = 0.097803 * np.tanh(np.where(data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"] < 0, np.maximum(((((data["ORGANIZATION_TYPE_Legal_Services"]) - (data["NAME_CONTRACT_TYPE_Revolving_loans"])))), ((
        data["ORGANIZATION_TYPE_Legal_Services"]))), np.maximum((((((data["CLOSED_AMT_ANNUITY_MEAN"]) > (data["ORGANIZATION_TYPE_Legal_Services"])) * 1.))), (((((data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]) > (3.141593)) * 1.))))))
    v["i137"] = 0.099961 * np.tanh((((((data["ORGANIZATION_TYPE_Mobile"]) - (data["NAME_INCOME_TYPE_Student"]))) + ((((((data["OCCUPATION_TYPE_Secretaries"]) - (data["NAME_EDUCATION_TYPE_Academic_degree"]))) + (
        ((data["ORGANIZATION_TYPE_Mobile"]) * (np.where(data["OCCUPATION_TYPE_Core_staff"] > 0, data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"], data["ORGANIZATION_TYPE_Mobile"]))))) / 2.0))) / 2.0))
    v["i138"] = 0.099620 * np.tanh(np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] > 0, np.minimum(((data["AMT_ANNUITY"])), ((0.0))), np.where(
        data["ORGANIZATION_TYPE_Kindergarten"] > 0, data["NEW_ANNUITY_TO_INCOME_RATIO"], (((data["AMT_ANNUITY"]) < (((-2.0) - (np.tanh((data["AMT_ANNUITY"])))))) * 1.))))
    v["i139"] = 0.093062 * np.tanh(((np.where(data["NAME_INCOME_TYPE_Working"] < 0, np.where(data["AMT_CREDIT"] > 0, data["NAME_CONTRACT_TYPE_Cash_loans"],
                                                                                             0.636620), (-1.0 * (((((data["AMT_CREDIT"]) > (np.minimum(((np.tanh((data["NEW_ANNUITY_TO_INCOME_RATIO"])))), ((0.636620))))) * 1.)))))) / 2.0))
    v["i140"] = 0.099801 * np.tanh(((((np.tanh(((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (-1.0)) * 1.)))) - (
        ((((((((-1.0 * ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) / 2.0)) / 2.0)) > (data["NAME_CONTRACT_TYPE_Cash_loans"])) * 1.)))) * 2.0))
    v["i141"] = 0.089801 * np.tanh(np.maximum(((data["BURO_STATUS_4_MEAN_MEAN"])), ((np.where((((2.0) + (data["NEW_DOC_IND_KURT"])) / 2.0) > 0, np.minimum(
        ((data["NEW_DOC_IND_KURT"])), ((0.0))), ((((-1.0 * ((data["BURO_STATUS_4_MEAN_MEAN"])))) > (data["NAME_CONTRACT_TYPE_Cash_loans"])) * 1.))))))
    v["i142"] = 0.062967 * np.tanh(np.where(data["BURO_STATUS_X_MEAN_MEAN"] > 0, np.where(((data["BURO_STATUS_4_MEAN_MEAN"]) + (data["BURO_STATUS_X_MEAN_MEAN"])) > 0, ((data["BURO_STATUS_4_MEAN_MEAN"]) - (
        ((np.tanh((data["BURO_STATUS_X_MEAN_MEAN"]))) + (data["BURO_MONTHS_BALANCE_MIN_MIN"])))), data["BURO_MONTHS_BALANCE_MIN_MIN"]), data["NAME_INCOME_TYPE_Maternity_leave"]))
    v["i143"] = 0.099749 * np.tanh(np.where(data["INSTAL_PAYMENT_DIFF_VAR"] < -99998, (((data["FLAG_PHONE"]) < (0.636620)) * 1.), np.where(
        data["INSTAL_PAYMENT_DIFF_VAR"] < 0, (((0.636620) < (data["INSTAL_AMT_PAYMENT_STD"])) * 1.), np.where(data["INSTAL_AMT_PAYMENT_STD"] < 0, 1.0, -2.0))))
    v["i144"] = 0.099950 * np.tanh(np.where((-1.0 * ((np.where(data["POS_SK_DPD_DEF_MAX"] < 0, data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"],
                                                               (((-1.0 * ((data["POS_SK_DPD_MEAN"])))) + (0.318310)))))) < 0, (10.0), (-1.0 * ((0.318310)))))
    v["i145"] = 0.099710 * np.tanh((((((data["ORGANIZATION_TYPE_Other"]) * (np.maximum(((np.maximum(((data["NAME_INCOME_TYPE_Student"])), ((data["NAME_INCOME_TYPE_Commercial_associate"]))))), ((
        data["FLAG_PHONE"])))))) + ((((np.maximum(((data["NAME_INCOME_TYPE_Maternity_leave"])), ((data["NAME_INCOME_TYPE_Commercial_associate"])))) < (data["FLAG_PHONE"])) * 1.))) / 2.0))
    v["i146"] = 0.099550 * np.tanh(((data["REFUSED_DAYS_DECISION_MEAN"]) * (((3.0) * ((((((np.tanh(((((data["REFUSED_DAYS_DECISION_MAX"]) > (
        0.318310)) * 1.)))) > (data["REFUSED_DAYS_DECISION_MEAN"])) * 1.)) - ((((0.636620) > (data["REFUSED_DAYS_DECISION_MAX"])) * 1.))))))))
    v["i147"] = 0.099951 * np.tanh(((np.tanh((((((((data["AMT_ANNUITY"]) * ((13.30679225921630859)))) * 2.0)) * 2.0)))) - (np.where(
        data["NAME_INCOME_TYPE_Student"] < -99998, data["EXT_SOURCE_2"], ((((data["EXT_SOURCE_2"]) + (data["EXT_SOURCE_2"]))) * 2.0)))))
    v["i148"] = 0.100000 * np.tanh(((((((data["HOUR_APPR_PROCESS_START"]) > (data["REGION_RATING_CLIENT"])) * 1.)) + (np.minimum(((np.where(
        data["LIVINGAREA_AVG"] < -99998, np.tanh((((data["HOUR_APPR_PROCESS_START"]) * 2.0))), (-1.0 * ((data["REGION_RATING_CLIENT"])))))), ((0.0))))) / 2.0))
    v["i149"] = 0.093000 * np.tanh(np.where((((1.570796) + (np.where(data["EXT_SOURCE_3"] < -99998, data["EXT_SOURCE_3"], data["EXT_SOURCE_1"]))) / 2.0) < 0, data[
                                   "ORGANIZATION_TYPE_Advertising"], np.where((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) + (1.570796)) / 2.0) < 0, data["EXT_SOURCE_1"], data["EXT_SOURCE_3"])))
    v["i150"] = 0.096966 * np.tanh((((np.minimum(((data["EXT_SOURCE_1"])), ((np.where(data["HOUR_APPR_PROCESS_START"] > 0, data["EXT_SOURCE_3"], data[
                                   "NAME_INCOME_TYPE_Maternity_leave"]))))) > ((((((0.318310) * (np.minimum(((-2.0)), ((data["EXT_SOURCE_3"])))))) + (data["EXT_SOURCE_1"])) / 2.0))) * 1.))
    v["i151"] = 0.094980 * np.tanh((((((6.0)) + (data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]))) * (((((6.0)) < (np.where(data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]
                                                                                                                     < 0, ((((data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]) - (-2.0))) * 2.0), data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]))) * 1.))))
    v["i152"] = 0.099961 * np.tanh((((((((3.0) < ((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) + (((((((np.tanh((data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"]))) + (
        1.0)) / 2.0)) + (np.maximum(((data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"])), ((data["CLOSED_CREDIT_DAY_OVERDUE_MEAN"]))))) / 2.0))) / 2.0))) * 1.)) * 2.0)) * 2.0))
    v["i153"] = 0.093200 * np.tanh((((-1.0 * ((((((((((data["HOUR_APPR_PROCESS_START"]) / 2.0)) + (data["HOUR_APPR_PROCESS_START"])) / 2.0)) > (
        (((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) > (((((data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]) * (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]))) - (0.318310)))) * 1.))) * 1.))))) / 2.0))
    v["i154"] = 0.100000 * np.tanh(((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (np.where(((((-1.0 * ((data["NEW_ANNUITY_TO_INCOME_RATIO"])))) < ((0.95379376411437988))) * 1.) > 0, data[
                                   "NAME_INCOME_TYPE_Maternity_leave"], np.where(data["DAYS_BIRTH"] > 0, 0.636620, (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (data["DAYS_BIRTH"])) / 2.0))))))
    v["i155"] = 0.099949 * np.tanh(((((data["AMT_INCOME_TOTAL"]) * 2.0)) * (np.maximum(((np.minimum(((((data["NEW_DOC_IND_KURT"]) * (
        data["NEW_DOC_IND_KURT"])))), ((data["NEW_DOC_IND_KURT"]))))), ((((data["BURO_STATUS_nan_MEAN_MEAN"]) - (((data["NEW_DOC_IND_KURT"]) * 2.0)))))))))
    v["i156"] = 0.099200 * np.tanh(((((data["ORGANIZATION_TYPE_Realtor"]) + (3.0))) * (((data["ORGANIZATION_TYPE_Realtor"]) + ((((data["ORGANIZATION_TYPE_Construction"]) > (
        np.where(data["EXT_SOURCE_3"] < -99998, data["ORGANIZATION_TYPE_Construction"], (((data["EXT_SOURCE_3"]) + (1.570796)) / 2.0)))) * 1.))))))
    v["i157"] = 0.099950 * np.tanh(((((data["BURO_MONTHS_BALANCE_SIZE_MEAN"]) * (np.maximum(((data["BURO_STATUS_4_MEAN_MEAN"])), ((
        (((((data["BURO_STATUS_0_MEAN_MEAN"]) > (2.0)) * 1.)) * (data["BURO_MONTHS_BALANCE_MIN_MIN"])))))))) * 2.0))
    v["i158"] = 0.082050 * np.tanh(((data["NEW_SCORES_STD"]) * ((((((((data["NEW_SCORES_STD"]) > (
        ((((data["NAME_TYPE_SUITE_Group_of_people"]) + (3.0))) - (0.318310)))) * 1.)) - (data["NAME_TYPE_SUITE_Group_of_people"]))) - (0.318310)))))
    v["i159"] = 0.099501 * np.tanh(np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < -99998, data["NEW_CREDIT_TO_ANNUITY_RATIO"], ((((((data["ORGANIZATION_TYPE_Telecom"]) + (
        ((data["ORGANIZATION_TYPE_Telecom"]) * 2.0))) / 2.0)) > ((((((((data["ORGANIZATION_TYPE_Telecom"]) * 2.0)) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) * 1.)))
    v["i160"] = 0.079550 * np.tanh((-1.0 * (((((data["REFUSED_AMT_APPLICATION_MEAN"]) > ((((((data["REFUSED_DAYS_DECISION_MAX"]) > (
        ((((data["REFUSED_HOUR_APPR_PROCESS_START_MAX"]) - (data["REFUSED_AMT_APPLICATION_MEAN"]))) - (((0.318310) * 2.0))))) * 1.)) * (data["REFUSED_AMT_CREDIT_MEAN"])))) * 1.)))))
    v["i161"] = 0.070000 * np.tanh((((((np.maximum(((((data["BURO_MONTHS_BALANCE_SIZE_MEAN"]) - (data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"])))), ((np.maximum(((((data["BURO_MONTHS_BALANCE_SIZE_MEAN"]) - (
        data["BURO_MONTHS_BALANCE_MAX_MAX"])))), ((np.minimum(((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"])), ((data["BURO_MONTHS_BALANCE_MAX_MAX"])))))))))) * 2.0)) < (data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"])) * 1.))
    v["i162"] = 0.098891 * np.tanh(np.where((((2.0) < (np.maximum(((data["BURO_STATUS_2_MEAN_MEAN"])), ((data["CLOSED_MONTHS_BALANCE_MIN_MIN"]))))) * 1.)
                                            > 0, 2.0, (-1.0 * (((((data["BURO_STATUS_2_MEAN_MEAN"]) > ((-1.0 * (((((data["CLOSED_MONTHS_BALANCE_MIN_MIN"]) > (0.318310)) * 1.)))))) * 1.))))))
    v["i163"] = 0.099819 * np.tanh(np.where(data["CLOSED_AMT_ANNUITY_MAX"] > 0, (((data["AMT_INCOME_TOTAL"]) > (0.318310)) * 1.), (((-2.0) > (np.where(
        data["EXT_SOURCE_3"] < -99998, data["NAME_INCOME_TYPE_Maternity_leave"], ((data["EXT_SOURCE_3"]) - (data["NAME_INCOME_TYPE_Maternity_leave"]))))) * 1.)))
    v["i164"] = 0.097050 * np.tanh((-1.0 * (((((((np.maximum(((0.318310)), ((((((1.570796) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) + (
        data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))) * (0.636620))) > (((1.570796) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) * 1.)))))
    v["i165"] = 0.099981 * np.tanh((-1.0 * ((((data["NAME_TYPE_SUITE_Group_of_people"]) + (((data["ORGANIZATION_TYPE_Trade__type_6"]) + (((data["ORGANIZATION_TYPE_Trade__type_4"]) + (np.where(
        data["CLOSED_AMT_ANNUITY_MEAN"] < -99998, 0.0, np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] > 0, data["ORGANIZATION_TYPE_Trade__type_6"], data["FONDKAPREMONT_MODE_reg_oper_spec_account"]))))))))))))
    v["i166"] = 0.022300 * np.tanh((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["NAME_EDUCATION_TYPE_Higher_education"]))) > (np.maximum(
        ((((data["NAME_EDUCATION_TYPE_Higher_education"]) / 2.0))), ((np.maximum(((np.tanh((data["REG_CITY_NOT_WORK_CITY"])))), ((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"])))))))) * 1.))
    v["i167"] = 0.087950 * np.tanh((((np.where(data["WALLSMATERIAL_MODE_Panel"] < 0, np.maximum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), ((data["NAME_INCOME_TYPE_Student"]))), ((
        data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0))) < (np.minimum(((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0))), (((-1.0 * ((((data["NAME_EDUCATION_TYPE_Academic_degree"]) * 2.0))))))))) * 1.))
    v["i168"] = 0.069600 * np.tanh((((np.where(data["NAME_EDUCATION_TYPE_Secondary___secondary_special"] > 0, data["ORGANIZATION_TYPE_Industry__type_4"], data["NEW_ANNUITY_TO_INCOME_RATIO"])) + (
        (((-1.0 * (((((np.where(data["NAME_EDUCATION_TYPE_Secondary___secondary_special"] < 0, (5.13249444961547852), data["NAME_EDUCATION_TYPE_Secondary___secondary_special"])) < (data["AMT_ANNUITY"])) * 1.))))) * (data["AMT_ANNUITY"])))) / 2.0))
    v["i169"] = 0.057497 * np.tanh((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (np.where(((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (
        (((-1.0) < (data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.))) < 0, ((-1.0) * (1.570796)), 1.570796))) * 1.)) * 2.0)) * 2.0))
    v["i170"] = 0.099700 * np.tanh(((((((((np.where(np.maximum(((data["WALLSMATERIAL_MODE_Panel"])), ((data["AMT_INCOME_TOTAL"]))) > 0, (-1.0 * ((data["AMT_INCOME_TOTAL"]))), (-1.0 * (
        (data["FONDKAPREMONT_MODE_reg_oper_account"]))))) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)) * (data["WALLSMATERIAL_MODE_Panel"])))
    v["i171"] = 0.098400 * np.tanh(np.where(((-1.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) < 0, ((
        (((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < ((-1.0 * (((((0.0) + (3.0)) / 2.0)))))) * 1.)) * 2.0)) * 2.0)) * 2.0), data["WALLSMATERIAL_MODE_Stone__brick"]))
    v["i172"] = 0.099968 * np.tanh(((np.minimum((((((((((0.318310) < ((((0.318310) < (data["NEW_CREDIT_TO_INCOME_RATIO"])) * 1.))) * 1.)) * (
        2.0))) - (data["NONLIVINGAREA_MODE"])))), (((((data["NONLIVINGAREA_AVG"]) > (data["NONLIVINGAREA_MODE"])) * 1.))))) * 2.0))
    v["i173"] = 0.090999 * np.tanh(np.where(data["NONLIVINGAREA_AVG"] < 0, (((((((((data["AMT_CREDIT"]) > ((2.85303425788879395))) * 1.)) * 2.0)) * 2.0)) * (
        data["NEW_CREDIT_TO_INCOME_RATIO"])), ((((((data["NAME_INCOME_TYPE_Student"]) > (data["NEW_CREDIT_TO_INCOME_RATIO"])) * 1.)) > (data["AMT_CREDIT"])) * 1.)))
    v["i174"] = 0.099500 * np.tanh((((-1.0 * (((((data["YEARS_BUILD_MEDI"]) > ((((np.where(data["YEARS_BUILD_MEDI"] > 0, (((1.0) > (
        ((data["NEW_CREDIT_TO_INCOME_RATIO"]) + (1.570796)))) * 1.), data["LIVINGAPARTMENTS_MODE"])) + (1.570796)) / 2.0))) * 1.))))) * 2.0))
    v["i175"] = 0.099504 * np.tanh(np.where(data["NONLIVINGAREA_MEDI"] < -99998, data["NAME_INCOME_TYPE_Maternity_leave"], np.where(((data["AMT_CREDIT"]) - (
        data["NONLIVINGAREA_MEDI"])) < 0, data["NAME_INCOME_TYPE_Maternity_leave"], (((((((data["NEW_CREDIT_TO_INCOME_RATIO"]) > (((3.0) / 2.0))) * 1.)) * 2.0)) * 2.0))))
    v["i176"] = 0.094380 * np.tanh(((((data["LIVINGAPARTMENTS_MODE"]) * ((((data["TOTALAREA_MODE"]) < ((3.0))) * 1.)))) * ((((np.minimum(
        ((data["TOTALAREA_MODE"])), ((1.570796)))) < (np.where(data["NONLIVINGAREA_MODE"] < -99998, data["TOTALAREA_MODE"], data["LIVINGAPARTMENTS_AVG"]))) * 1.))))
    v["i177"] = 0.099900 * np.tanh((((-1.0 * (((((data["TOTALAREA_MODE"]) > (np.where(np.where(data["LIVINGAPARTMENTS_MODE"] < -99998, data["TOTALAREA_MODE"], ((
        (((data["EMERGENCYSTATE_MODE_No"]) * 2.0)) < (data["NONLIVINGAREA_MEDI"])) * 1.)) > 0, data["EMERGENCYSTATE_MODE_No"], 3.0))) * 1.))))) * 2.0))
    v["i178"] = 0.099700 * np.tanh((((((((((data["AMT_INCOME_TOTAL"]) > ((((((((5.40422677993774414)) + (np.where(data["TOTALAREA_MODE"] < -99998, (4.0), data[
                                   "NONLIVINGAREA_MEDI"]))) / 2.0)) + (data["AMT_INCOME_TOTAL"])) / 2.0))) * 1.)) * (data["AMT_INCOME_TOTAL"]))) * 2.0)) * 2.0))
    v["i179"] = 0.050498 * np.tanh(((np.where(np.minimum(((data["LIVINGAPARTMENTS_MEDI"])), ((data["NONLIVINGAREA_MODE"]))) < -99998, (((-1.0) < (data["LIVINGAPARTMENTS_MEDI"])) * 1.), data[
                                   "NONLIVINGAPARTMENTS_MEDI"])) * ((((data["NONLIVINGAREA_MODE"]) < ((((data["NONLIVINGAREA_AVG"]) > (data["LIVINGAPARTMENTS_MODE"])) * 1.))) * 1.))))
    v["i180"] = 0.019700 * np.tanh((((((((((np.minimum(((data["NONLIVINGAREA_MODE"])), ((1.570796)))) + (3.0))) < (
        data["LIVINGAPARTMENTS_MEDI"])) * 1.)) * (((((1.570796) + (data["LIVINGAPARTMENTS_MEDI"]))) + (data["LIVINGAPARTMENTS_MODE"]))))) * 2.0))
    v["i181"] = 0.099496 * np.tanh((((np.maximum(((np.where(((((((0.318310) < (data["LIVINGAPARTMENTS_AVG"])) * 1.)) > (data["LIVINGAPARTMENTS_AVG"])) * 1.)
                                                            > 0, data["NONLIVINGAREA_AVG"], -2.0))), ((((data["LIVINGAPARTMENTS_AVG"]) - (data["NONLIVINGAREA_MODE"])))))) > (1.570796)) * 1.))
    v["i182"] = 0.078968 * np.tanh(((np.where((((data["LIVINGAPARTMENTS_AVG"]) > (data["LIVINGAPARTMENTS_MEDI"])) * 1.) > 0, (((((((data["LIVINGAPARTMENTS_MEDI"]) > (
        data["LIVINGAPARTMENTS_MODE"])) * 1.)) - (data["LIVINGAPARTMENTS_MODE"]))) - (3.0)), data["LIVINGAPARTMENTS_MEDI"])) - (data["LIVINGAPARTMENTS_MODE"])))
    v["i183"] = 0.098001 * np.tanh((((((data["NONLIVINGAREA_MEDI"]) < (((data["NAME_INCOME_TYPE_Student"]) + (((data["NAME_INCOME_TYPE_Student"]) + (
        data["NONLIVINGAREA_AVG"])))))) * 1.)) - ((((np.maximum(((data["NONLIVINGAREA_AVG"])), ((3.141593)))) < (data["NONLIVINGAREA_MEDI"])) * 1.))))
    v["i184"] = 0.048400 * np.tanh((((((((np.minimum(((np.minimum(((data["NONLIVINGAREA_AVG"])), (((-1.0 * ((data["NONLIVINGAREA_MODE"])))))))), ((
        ((((np.tanh((data["EMERGENCYSTATE_MODE_No"]))) * 2.0)) / 2.0))))) * (3.141593))) > (data["NONLIVINGAREA_MEDI"])) * 1.)) * 2.0))
    v["i185"] = 0.088029 * np.tanh((-1.0 * (((((((data["NONLIVINGAPARTMENTS_AVG"]) < (data["LIVINGAPARTMENTS_MODE"])) * 1.)) * (
        (((((((data["NONLIVINGAPARTMENTS_AVG"]) + (data["NONLIVINGAREA_MODE"])) / 2.0)) * (data["NONLIVINGAREA_MODE"]))) - (data["NONLIVINGAREA_MODE"]))))))))
    v["i186"] = 0.099699 * np.tanh(((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) * ((((((data["FONDKAPREMONT_MODE_not_specified"]) + (
        data["FONDKAPREMONT_MODE_not_specified"]))) > ((((data["FONDKAPREMONT_MODE_not_specified"]) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (np.tanh((-2.0)))))) / 2.0))) * 1.))))
    v["i187"] = 0.053159 * np.tanh(((np.where((((data["LIVINGAPARTMENTS_AVG"]) > (data["LIVINGAPARTMENTS_MEDI"])) * 1.) > 0, ((
        data["LIVINGAPARTMENTS_MODE"]) - ((5.28531932830810547))), (((-1.0 * ((np.tanh((np.tanh((data["NONLIVINGAREA_MODE"])))))))) / 2.0))) / 2.0))
    v["i188"] = 0.097900 * np.tanh((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < ((-1.0 * ((((np.where((((((data["NAME_EDUCATION_TYPE_Academic_degree"]) / 2.0)) > (
        data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.) > 0, data["NAME_INCOME_TYPE_Student"], ((-1.0) / 2.0))) - ((-1.0 * ((1.570796)))))))))) * 1.))
    v["i189"] = 0.085203 * np.tanh(np.where(data["DAYS_BIRTH"] < 0, np.where(data["AMT_ANNUITY"] > 0, ((((data["FONDKAPREMONT_MODE_not_specified"]) * (data["AMT_ANNUITY"]))) * (
        data["AMT_ANNUITY"])), (((data["DAYS_BIRTH"]) > (((data["FONDKAPREMONT_MODE_not_specified"]) / 2.0))) * 1.)), data["ORGANIZATION_TYPE_Legal_Services"]))
    v["i190"] = 0.100000 * np.tanh((((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (0.636620)) * 1.)) * 2.0)) - (0.318310))) * (
        (-1.0 * ((np.tanh((np.minimum(((data["DAYS_BIRTH"])), (((((data["DAYS_BIRTH"]) > (0.636620)) * 1.))))))))))))
    v["i191"] = 0.099503 * np.tanh((-1.0 * ((((((((((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < -99998, data["EXT_SOURCE_3"], 1.570796)) + (
        data["AMT_ANNUITY"])) / 2.0)) > (2.0)) * 1.)) * (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, (5.0), -2.0)))))))
    v["i192"] = 0.099943 * np.tanh(np.where(((0.636620) - (data["EXT_SOURCE_3"])) < 0, data["OCCUPATION_TYPE_Accountants"], np.where(data["EXT_SOURCE_3"] < -99998, np.where(
        data["AMT_INCOME_TOTAL"] < 0, data["AMT_INCOME_TOTAL"], (((data["AMT_ANNUITY"]) < (data["ORGANIZATION_TYPE_Advertising"])) * 1.)), data["ORGANIZATION_TYPE_Advertising"])))
    v["i193"] = 0.099000 * np.tanh((-1.0 * ((((((((((((((((((-1.0 * ((data["NAME_EDUCATION_TYPE_Academic_degree"])))) - (data["ORGANIZATION_TYPE_Transport__type_1"])))
                                                         * 2.0)) - (-2.0))) - (data["NAME_INCOME_TYPE_Maternity_leave"]))) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)) * 2.0)) * 2.0)))))
    v["i194"] = 0.099590 * np.tanh(np.where(data["WALLSMATERIAL_MODE_Others"] < 0, ((((-1.0 * ((data["WALLSMATERIAL_MODE_Others"])))) > ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (
        ((((((-1.0 * ((data["WALLSMATERIAL_MODE_Others"])))) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)) * 2.0))) * 1.))) * 1.), data["AMT_ANNUITY"]))
    v["i195"] = 0.070396 * np.tanh((((((1.570796) * ((((((data["DAYS_BIRTH"]) + (1.570796))) > (((3.0) / 2.0))) * 1.)))) > (
        np.maximum(((((data["DAYS_BIRTH"]) + (1.570796)))), ((data["CLOSED_AMT_ANNUITY_MEAN"]))))) * 1.))
    v["i196"] = 0.084510 * np.tanh(np.where(np.maximum(((data["OCCUPATION_TYPE_Sales_staff"])), ((((data["CLOSED_AMT_ANNUITY_MAX"]) / 2.0)))) < 0, data[
                                   "ORGANIZATION_TYPE_Trade__type_2"], np.maximum(((((data["AMT_ANNUITY"]) * 2.0))), ((data["ORGANIZATION_TYPE_Trade__type_2"])))))
    v["i197"] = 0.098240 * np.tanh(((((((data["WALLSMATERIAL_MODE_Others"]) + (np.where(data["EXT_SOURCE_3"] > 0, np.maximum(((data["ORGANIZATION_TYPE_Housing"])), ((
        data["WALLSMATERIAL_MODE_Others"]))), (((data["ORGANIZATION_TYPE_Housing"]) < (data["EXT_SOURCE_3"])) * 1.))))) * (((data["AMT_ANNUITY"]) * 2.0)))) / 2.0))
    v["i198"] = 0.099946 * np.tanh(((data["WALLSMATERIAL_MODE_Others"]) + ((((data["NAME_INCOME_TYPE_Student"]) < (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, data[
                                   "WALLSMATERIAL_MODE_Others"], ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["ORGANIZATION_TYPE_Industry__type_3"]))))) * 1.))))
    v["i199"] = 0.089500 * np.tanh((((((((((3.0) / 2.0)) < ((-1.0 * ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))) * 1.)) * 2.0)) - (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, data[
                                   "ORGANIZATION_TYPE_Industry__type_12"], (((-1.0 * ((data["ORGANIZATION_TYPE_Industry__type_12"])))) - (data["ORGANIZATION_TYPE_Industry__type_12"]))))))
    v["i200"] = 0.079602 * np.tanh(np.minimum(((((1.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))), ((((((((data["AMT_ANNUITY"]) > (
        data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)) < (((((((data["AMT_ANNUITY"]) > (data["EXT_SOURCE_3"])) * 1.)) < (((data["AMT_ANNUITY"]) * 2.0))) * 1.))) * 1.)))))
    v["i201"] = 0.090045 * np.tanh(np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] > 0, ((-1.0) + ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (
        ((2.0) + (data["WALLSMATERIAL_MODE_Mixed"])))) * 1.))), (-1.0 * ((data["ORGANIZATION_TYPE_Industry__type_12"])))))
    v["i202"] = 0.097199 * np.tanh((-1.0 * ((np.where(data["INSTAL_DAYS_ENTRY_PAYMENT_STD"] < -99998, data["NEW_ANNUITY_TO_INCOME_RATIO"], ((0.636620) - (
        (((((np.minimum(((np.tanh((np.tanh((0.318310)))))), ((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"])))) * 2.0)) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.))))))))
    v["i203"] = 0.068270 * np.tanh((((((np.where(data["CC_NAME_CONTRACT_STATUS_Active_SUM"] < 0, (-1.0 * ((data["CC_NAME_CONTRACT_STATUS_Refused_MEAN"]))), np.minimum(((data["CC_SK_DPD_DEF_MEAN"])),
                                                                                                                                                                       ((data["CC_NAME_CONTRACT_STATUS_Active_SUM"]))))) < (np.where(data["CC_COUNT"] > 0, data["CC_NAME_CONTRACT_STATUS_Refused_MEAN"], data["CC_SK_DPD_DEF_MEAN"]))) * 1.)) * 2.0))
    v["i204"] = 0.098990 * np.tanh(np.where(data["CLOSED_AMT_ANNUITY_MEAN"] < -99998, (((-1.0 * (((((((data["EXT_SOURCE_3"]) < (
        ((((0.636620) / 2.0)) + (data["AMT_ANNUITY"])))) * 1.)) / 2.0))))) / 2.0), (((0.636620) < (data["AMT_ANNUITY"])) * 1.)))
    v["i205"] = 0.030500 * np.tanh((((-1.0 * ((np.where(data["BURO_STATUS_4_MEAN_MEAN"] > 0, data["BURO_STATUS_C_MEAN_MEAN"], (((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]) > (
        ((1.0) - (np.where(data["CLOSED_MONTHS_BALANCE_MAX_MAX"] > 0, data["AMT_CREDIT"], data["BURO_STATUS_4_MEAN_MEAN"]))))) * 1.)))))) * 2.0))
    v["i206"] = 0.068540 * np.tanh(np.where((((((((-1.0) + ((((data["CLOSED_MONTHS_BALANCE_MAX_MAX"]) + (data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"])) / 2.0))) / 2.0)) - (np.minimum(
        ((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"])))))) * 2.0) < 0, data["CLOSED_MONTHS_BALANCE_MIN_MIN"], data["NAME_INCOME_TYPE_Maternity_leave"]))
    v["i207"] = 0.099171 * np.tanh(((((((((-1.0) < ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (np.tanh((np.tanh((-1.0)))))) / 2.0)))
                                         * 1.)) / 2.0)) > ((((3.141593) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0))) / 2.0))) * 1.))
    v["i208"] = 0.030500 * np.tanh((-1.0 * (((((((np.where(data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"] < 0, data["BURO_STATUS_C_MEAN_MEAN"], np.tanh((np.maximum(((data["CLOSED_MONTHS_BALANCE_MAX_MAX"])),
                                                                                                                                                             ((data["BURO_STATUS_X_MEAN_MEAN"]))))))) > (np.minimum(((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"])), ((data["CLOSED_MONTHS_BALANCE_MAX_MAX"]))))) * 1.)) / 2.0)))))
    v["i209"] = 0.091100 * np.tanh(np.where(data["NEW_DOC_IND_KURT"] > 0, (-1.0 * (
        ((((((2.0) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) < (data["NEW_DOC_IND_KURT"])) * 1.)))), data["NEW_CREDIT_TO_ANNUITY_RATIO"]))
    v["i210"] = 0.069001 * np.tanh(np.tanh((np.tanh(((((data["DAYS_BIRTH"]) < (((np.tanh((((((0.37940034270286560)) > (data["AMT_ANNUITY"])) * 1.)))) - (
        np.where(data["CC_NAME_CONTRACT_STATUS_Refused_SUM"] < -99998, (2.0), data["DAYS_BIRTH"]))))) * 1.))))))
    v["i211"] = 0.099939 * np.tanh((((np.minimum(((data["CNT_CHILDREN"])), ((((((data["CNT_CHILDREN"]) * (((data["CNT_CHILDREN"]) * (
        data["AMT_ANNUITY"]))))) + ((((0.318310) > (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0))) * 1.))))))) + (0.318310)) / 2.0))
    v["i212"] = 0.099979 * np.tanh((((((((data["NAME_INCOME_TYPE_Maternity_leave"]) * (data["NAME_CONTRACT_TYPE_Cash_loans"]))) + (np.minimum(((((np.where(data["NEW_DOC_IND_KURT"] < 0, data[
                                   "NAME_CONTRACT_TYPE_Revolving_loans"], data["NAME_CONTRACT_TYPE_Cash_loans"])) / 2.0))), (((((data["NEW_DOC_IND_KURT"]) < (data["NAME_INCOME_TYPE_Maternity_leave"])) * 1.)))))) / 2.0)) / 2.0))
    v["i213"] = 0.096700 * np.tanh(((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]) * (((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)) * (np.where(data["INSTAL_PAYMENT_DIFF_STD"] < -99998, data[
                                   "CLOSED_MONTHS_BALANCE_SIZE_MEAN"], (((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]) < (np.tanh((((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]) + (data["INSTAL_DAYS_ENTRY_PAYMENT_STD"])))))) * 1.)))))))
    v["i214"] = 0.099035 * np.tanh((-1.0 * (((((((0.318310) < ((((data["DAYS_BIRTH"]) + ((((data["DAYS_BIRTH"]) < (
        (((((data["LIVE_REGION_NOT_WORK_REGION"]) + ((((data["DAYS_BIRTH"]) + (data["LIVE_REGION_NOT_WORK_REGION"])) / 2.0))) / 2.0)) / 2.0))) * 1.))) / 2.0))) * 1.)) / 2.0)))))
    v["i215"] = 0.095997 * np.tanh((((np.where(data["FLAG_WORK_PHONE"] > 0, data["DAYS_BIRTH"], (((np.tanh((data["DAYS_BIRTH"]))) > (
        (((0.318310) + (data["DAYS_BIRTH"])) / 2.0))) * 1.))) + ((((((data["DAYS_BIRTH"]) < (data["NEW_LIVE_IND_SUM"])) * 1.)) / 2.0))) / 2.0))
    v["i216"] = 0.098960 * np.tanh((((0.636620) + (np.minimum(((-1.0)), ((((np.where(data["CLOSED_MONTHS_BALANCE_MIN_MIN"] < -99998, data["NEW_CREDIT_TO_ANNUITY_RATIO"],
                                                                                     ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["NEW_ANNUITY_TO_INCOME_RATIO"])))) * (data["NEW_ANNUITY_TO_INCOME_RATIO"]))))))) / 2.0))
    v["i217"] = 0.099400 * np.tanh((-1.0 * (((((((((((data["CLOSED_MONTHS_BALANCE_MAX_MAX"]) + (data["NEW_ANNUITY_TO_INCOME_RATIO"])) / 2.0)) + (data["AMT_ANNUITY"])) / 2.0)) < (
        np.minimum((((-1.0 * (((((data["CLOSED_MONTHS_BALANCE_MAX_MAX"]) < (data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"])) * 1.)))))), ((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]))))) * 1.)))))
    v["i218"] = 0.096740 * np.tanh((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (((((((6.85217618942260742)) / 2.0)) + (np.where(
        data["AMT_ANNUITY"] > 0, np.maximum(((data["AMT_ANNUITY"])), ((0.636620))), ((data["FLAG_CONT_MOBILE"]) * 2.0)))) / 2.0))) * 1.))
    v["i219"] = 0.099379 * np.tanh((((((3.0) < (((np.where(data["NONLIVINGAREA_MEDI"] > 0, data["NONLIVINGAREA_MODE"], (-1.0 * (
        (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))) * 2.0))) * 1.)) * ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (((data["AMT_ANNUITY"]) * 2.0))) * 1.))))
    v["i220"] = 0.100000 * np.tanh(((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)) * 2.0)) * 2.0)) * ((((data["AMT_ANNUITY"]) > (
        (((((data["NONLIVINGAREA_AVG"]) < ((-1.0 * ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)))))) * 1.)) + (1.570796)))) * 1.))))
    v["i221"] = 0.099400 * np.tanh((-1.0 * (((((((((((5.76207304000854492)) > (data["NONLIVINGAREA_AVG"])) * 1.)) < (np.where(
        data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, 1.0, ((np.tanh(((((-1.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) / 2.0)))) * 2.0)))) * 1.)) * 2.0)))))
    v["i222"] = 0.0 * np.tanh(((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (np.where(data["NONLIVINGAREA_MEDI"] < 0, (((2.0) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.),
                                                                                 ((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0))) * 1.)) > (((data["NONLIVINGAREA_MODE"]) * 2.0))) * 1.)))))
    v["i223"] = 0.097100 * np.tanh((((((((((((np.where(np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] > 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], data["NONLIVINGAREA_MODE"]) > 0, data[
                                   "NEW_CREDIT_TO_ANNUITY_RATIO"], data["NONLIVINGAREA_AVG"])) > (2.0)) * 1.)) * 2.0)) * 2.0)) * 2.0)) - (data["NAME_INCOME_TYPE_Student"])))
    v["i224"] = 0.100000 * np.tanh((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (data["NONLIVINGAREA_MODE"])) / 2.0)) * ((((((np.maximum((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (
        data["NONLIVINGAREA_MODE"])) / 2.0))), ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) / 2.0))))) - (3.141593))) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.))))
    v["i225"] = 0.098097 * np.tanh(((((np.where(np.where(data["NONLIVINGAREA_MODE"] < 0, data["NAME_INCOME_TYPE_Maternity_leave"], data["NEW_ANNUITY_TO_INCOME_RATIO"])
                                                < 0, data["NONLIVINGAREA_MEDI"], 0.318310)) - (data["NONLIVINGAREA_MODE"]))) * (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * 2.0))))
    v["i226"] = 0.091960 * np.tanh(np.where(((np.where((-1.0 * ((data["NONLIVINGAREA_MEDI"]))) < 0, (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.), data[
                                   "NONLIVINGAREA_MEDI"])) - (data["NONLIVINGAREA_MODE"])) > 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], ((data["NONLIVINGAREA_MODE"]) - (data["NONLIVINGAREA_AVG"]))))
    v["i227"] = 0.099197 * np.tanh(((((data["AMT_ANNUITY"]) * (data["ORGANIZATION_TYPE_Housing"]))) - (np.maximum(((data["OCCUPATION_TYPE_IT_staff"])), ((
        np.minimum(((0.318310)), ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * ((((data["ORGANIZATION_TYPE_Housing"]) > (((data["AMT_ANNUITY"]) / 2.0))) * 1.))))))))))))
    v["i228"] = 0.089500 * np.tanh((((np.maximum(((3.141593)), ((data["NEW_ANNUITY_TO_INCOME_RATIO"])))) < (((data["AMT_ANNUITY"]) * (((np.where(
        ((data["AMT_ANNUITY"]) - (data["NEW_ANNUITY_TO_INCOME_RATIO"])) > 0, 1.570796, data["NEW_ANNUITY_TO_INCOME_RATIO"])) - (1.570796)))))) * 1.))
    v["i229"] = 0.091950 * np.tanh((((((np.maximum(((((((3.0) - ((((data["AMT_ANNUITY"]) < (np.tanh((np.tanh((data["NEW_CREDIT_TO_INCOME_RATIO"])))))) * 1.)))) + (
        data["AMT_ANNUITY"])))), ((1.570796)))) < (data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.)) * 2.0))
    v["i230"] = 0.099930 * np.tanh((-1.0 * ((((((((data["CNT_CHILDREN"]) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)) < ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (
        (((data["CNT_CHILDREN"]) + ((((data["ORGANIZATION_TYPE_Trade__type_4"]) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) / 2.0))) / 2.0))) * 1.))) * 1.)))))
    v["i231"] = 0.038000 * np.tanh((((np.where(np.minimum((((((0.636620) + (data["NEW_INC_PER_CHLD"])) / 2.0))), ((
        (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (((data["AMT_ANNUITY"]) / 2.0))) / 2.0)))) < 0, 1.570796, 3.141593)) < (data["AMT_ANNUITY"])) * 1.))
    v["i232"] = 0.100000 * np.tanh(((((data["NEW_INC_PER_CHLD"]) * ((((data["CNT_CHILDREN"]) > (2.0)) * 1.)))) + (
        (((((((2.0) < ((((data["CNT_CHILDREN"]) + (data["NEW_INC_PER_CHLD"])) / 2.0))) * 1.)) * 2.0)) * 2.0))))
    v["i233"] = 0.074001 * np.tanh((-1.0 * (((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (((0.636620) - ((((((5.0)) - (data["NAME_INCOME_TYPE_Student"]))) - (
        np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, 3.0, ((data["NONLIVINGAREA_MODE"]) * 2.0)))))))) * 1.)))))
    v["i234"] = 0.079999 * np.tanh(((((np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] < -99998, data["ORGANIZATION_TYPE_Hotel"], ((np.tanh(
        (np.tanh((((((data["AMT_ANNUITY"]) + (data["NONLIVINGAREA_AVG"]))) * 2.0)))))) * (data["ORGANIZATION_TYPE_Hotel"])))) * 2.0)) * 2.0))
    v["i235"] = 0.078003 * np.tanh(((np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] > 0, ((((-1.0 * ((0.636620)))) > (np.tanh(
        (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) * 1.), ((data["WALLSMATERIAL_MODE_Mixed"]) + (data["ORGANIZATION_TYPE_Business_Entity_Type_1"])))) * (0.636620)))
    v["i236"] = 0.075100 * np.tanh((((np.where((((data["AMT_ANNUITY"]) > (0.636620)) * 1.) > 0, ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])
                                                                                                 * 2.0), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) > (3.141593)) * 1.))
    v["i237"] = 0.099651 * np.tanh((-1.0 * (((((1.570796) < (((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (data["NAME_INCOME_TYPE_Student"]))) + (
        (((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["NEW_ANNUITY_TO_INCOME_RATIO"])))) * 1.)) / 2.0)) / 2.0))))) * 1.)))))
    v["i238"] = 0.044000 * np.tanh((((np.where(((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (data["NEW_CREDIT_TO_INCOME_RATIO"])) * 1.))) < 0, data[
                                   "NEW_CREDIT_TO_INCOME_RATIO"], (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (data["NEW_CREDIT_TO_INCOME_RATIO"])) * 1.))) > ((((1.0) > (data["REG_CITY_NOT_WORK_CITY"])) * 1.))) * 1.))
    v["i239"] = 0.090029 * np.tanh(((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * ((((((((((((3.141593) < (data["AMT_ANNUITY"])) * 1.)) * 2.0)) * 2.0))
                                                                                    * 2.0)) - (data["ORGANIZATION_TYPE_Trade__type_4"]))))) * 2.0)) - (data["ORGANIZATION_TYPE_Trade__type_4"])))
    v["i240"] = 0.090000 * np.tanh(((np.where(((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["NAME_EDUCATION_TYPE_Academic_degree"])) < 0, ((data["AMT_ANNUITY"]) * (
        ((((data["NAME_EDUCATION_TYPE_Academic_degree"]) * 2.0)) * 2.0))), ((((data["NAME_EDUCATION_TYPE_Academic_degree"]) * 2.0)) * 2.0))) * (data["REG_CITY_NOT_WORK_CITY"])))
    v["i241"] = 0.082502 * np.tanh(((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, 2.0, data["NAME_EDUCATION_TYPE_Academic_degree"])) * (
        (((-1.0 * ((data["NAME_EDUCATION_TYPE_Academic_degree"])))) + ((-1.0 * (((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (((data["NAME_EDUCATION_TYPE_Academic_degree"]) * 2.0))) * 1.)))))))))
    v["i242"] = 0.081500 * np.tanh(np.where(data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"] > 0, np.maximum((((((data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"]) < (1.570796)) * 1.))),
                                                                                                                        (((-1.0 * ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))), ((((((3.0) + ((-1.0 * ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))) / 2.0)) < (data["POS_NAME_CONTRACT_STATUS_Signed_MEAN"])) * 1.)))
    v["i243"] = 0.099150 * np.tanh(np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] > 0, data["HOUSETYPE_MODE_specific_housing"], ((np.where(-1.0 > 0, data[
                                   "HOUSETYPE_MODE_specific_housing"], ((-1.0) + ((((data["DAYS_BIRTH"]) < (((3.0) / 2.0))) * 1.))))) - (data["NAME_HOUSING_TYPE_Office_apartment"]))))
    v["i244"] = 0.099990 * np.tanh(np.where(data["OCCUPATION_TYPE_Core_staff"] > 0, (((((data["DAYS_ID_PUBLISH"]) * (data["AMT_ANNUITY"]))) < (
        data["AMT_ANNUITY"])) * 1.), (-1.0 * (((((((-1.0) > ((((data["DAYS_ID_PUBLISH"]) + (data["AMT_ANNUITY"])) / 2.0))) * 1.)) * 2.0))))))
    v["i245"] = 0.059010 * np.tanh((((((((-1.0 * (((((data["ORGANIZATION_TYPE_Trade__type_4"]) > ((((((((((data["ORGANIZATION_TYPE_Bank"]) / 2.0)) * (
        data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) / 2.0)) * (data["ORGANIZATION_TYPE_Trade__type_4"])))) * 1.))))) * 2.0)) * 2.0)) * 2.0))
    v["i246"] = 0.099998 * np.tanh((((((((((((-1.0 * ((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0))) / 2.0)) > (
        1.570796)) * 1.))))) * 2.0)) - (data["NAME_INCOME_TYPE_Student"]))) * 2.0)) * 2.0)) - (data["ORGANIZATION_TYPE_Trade__type_4"])))
    v["i247"] = 0.050000 * np.tanh(((((data["INSTAL_AMT_PAYMENT_STD"]) * ((((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]) > ((((data["INSTAL_AMT_PAYMENT_STD"]) < (2.0)) * 1.))) * 1.)))) + (
        (((np.where(data["INSTAL_AMT_PAYMENT_STD"] < -99998, data["INSTAL_AMT_PAYMENT_STD"], data["NEW_CREDIT_TO_ANNUITY_RATIO"])) > (2.0)) * 1.))))
    v["i248"] = 0.085002 * np.tanh(np.minimum(((np.minimum(((((data["INSTAL_PAYMENT_DIFF_STD"]) * (data["INSTAL_DAYS_ENTRY_PAYMENT_STD"])))), ((((data["INSTAL_PAYMENT_DIFF_STD"]) - (
        data["INSTAL_PAYMENT_DIFF_VAR"]))))))), ((np.where(data["INSTAL_PAYMENT_PERC_STD"] < 0, (-1.0 * ((data["INSTAL_PAYMENT_DIFF_VAR"]))), data["INSTAL_AMT_PAYMENT_STD"])))))
    v["i249"] = 0.092700 * np.tanh(np.maximum(((np.where(data["REGION_RATING_CLIENT_W_CITY"] < 0, (((-1.0 * ((np.tanh((data["OCCUPATION_TYPE_Waiters_barmen_staff"])))))) * (
        data["REGION_RATING_CLIENT_W_CITY"])), (((data["NAME_HOUSING_TYPE_With_parents"]) + (data["OCCUPATION_TYPE_Waiters_barmen_staff"])) / 2.0)))), (((((((0.16719464957714081)) / 2.0)) / 2.0)))))
    v["i250"] = 0.099695 * np.tanh(((((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)) / 2.0)) > (data["ORGANIZATION_TYPE_Housing"])) * 1.))
                                     > ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (((data["ORGANIZATION_TYPE_Housing"]) / 2.0))) * 1.))) * 1.))
    v["i251"] = 0.097400 * np.tanh(np.minimum(((((data["NEW_DOC_IND_KURT"]) - (np.where(data["DAYS_BIRTH"] > 0, data["DAYS_REGISTRATION"], (-1.0 * (
        (((((data["DAYS_REGISTRATION"]) * ((-1.0 * ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))) * 2.0))))))))), ((data["NAME_CONTRACT_TYPE_Cash_loans"]))))
    v["i252"] = 0.099195 * np.tanh((((((((data["INSTAL_PAYMENT_PERC_STD"]) > (((data["INSTAL_PAYMENT_PERC_VAR"]) - (
        (((((0.636620) * 2.0)) < (data["DAYS_REGISTRATION"])) * 1.))))) * 1.)) * (((((((data["DAYS_REGISTRATION"]) * 2.0)) * 2.0)) * 2.0)))) * 2.0))
    v["i253"] = 0.098171 * np.tanh((((np.where(data["NEW_DOC_IND_KURT"] > 0, 2.0, np.where(((data["NEW_DOC_IND_KURT"]) + (
        3.0)) > 0, ((data["INSTAL_AMT_PAYMENT_STD"]) * 2.0), ((0.0) - (0.318310))))) < (data["INSTAL_AMT_PAYMENT_STD"])) * 1.))
    v["i254"] = 0.097303 * np.tanh(((np.where(data["INSTAL_PAYMENT_DIFF_VAR"] < 0, (-1.0 * ((data["NAME_INCOME_TYPE_Student"]))), np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] < 0, data[
                                   "NEW_CREDIT_TO_ANNUITY_RATIO"], np.where(data["INSTAL_DAYS_ENTRY_PAYMENT_STD"] < 0, (-1.0 * ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))), data["AMT_ANNUITY"])))) * (-2.0)))
    v["i255"] = 0.099660 * np.tanh((((((-1.0 * (((((np.tanh((1.570796))) < (np.where(data["CLOSED_MONTHS_BALANCE_MIN_MIN"] > 0, data["INSTAL_DAYS_ENTRY_PAYMENT_STD"], ((
        (np.minimum(((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"])), ((data["NEW_ANNUITY_TO_INCOME_RATIO"])))) > (data["INSTAL_DAYS_ENTRY_PAYMENT_STD"])) * 1.)))) * 1.))))) * 2.0)) * 2.0))
    v["i256"] = 0.099733 * np.tanh((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > ((((((0.0) < ((((1.0) + (np.maximum(((data["CLOSED_AMT_ANNUITY_MEAN"])), ((np.where(
        data["NEW_CREDIT_TO_INCOME_RATIO"] > 0, data["INSTAL_DAYS_ENTRY_PAYMENT_STD"], data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))) / 2.0))) * 1.)) * 2.0))) * 1.)) * 2.0))
    v["i257"] = 0.099308 * np.tanh(np.where(data["BURO_AMT_ANNUITY_MEAN"] < 0, np.where(data["CLOSED_MONTHS_BALANCE_MIN_MIN"] > 0, data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"], (((np.where(data["BURO_AMT_ANNUITY_MEAN"]
                                                                                                                                                                                        < -99998, data["BURO_AMT_ANNUITY_MAX"], ((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) * ((5.97321271896362305))))) > (data["BURO_AMT_ANNUITY_MEAN"])) * 1.)), data["CLOSED_AMT_ANNUITY_MAX"]))
    v["i258"] = 0.077200 * np.tanh(((((((((data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"]) + (data["NAME_INCOME_TYPE_Maternity_leave"])) / 2.0)) < (data["POS_SK_DPD_MEAN"])) * 1.)) + (
        np.where(data["BURO_AMT_ANNUITY_MEAN"] < -99998, np.tanh((data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"])), (((-1.0 * ((data["BURO_AMT_ANNUITY_MEAN"])))) - (data["POS_SK_DPD_MAX"]))))))
    v["i259"] = 0.098503 * np.tanh(((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (np.where(data["INSTAL_AMT_PAYMENT_STD"] < 0, ((np.where(data["INSTAL_PAYMENT_DIFF_VAR"] > 0, (6.0), np.where(
        data["NEW_ANNUITY_TO_INCOME_RATIO"] < 0, data["INSTAL_PAYMENT_DIFF_VAR"], data["INSTAL_PAYMENT_DIFF_STD"]))) / 2.0), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)))))
    v["i260"] = 0.099800 * np.tanh(((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - ((1.95137548446655273)))))) * (
        (((1.570796) < (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])) * 1.)))) - ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > ((1.95137548446655273))) * 1.)))) * 2.0))
    v["i261"] = 0.079810 * np.tanh((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (np.where(
        ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + ((-1.0 * ((0.636620))))) < 0, ((1.570796) + ((-1.0 * ((1.0))))), 2.0))) * 1.))
    v["i262"] = 0.097999 * np.tanh((-1.0 * ((np.where((((((-2.0) + (data["NEW_ANNUITY_TO_INCOME_RATIO"])) / 2.0)) / 2.0) > 0, ((
        (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"])) / 2.0)) / 2.0), ((((5.0)) < (data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"])) * 1.))))))
    v["i263"] = 0.098397 * np.tanh(((((np.where(data["OCCUPATION_TYPE_Accountants"] > 0, data["OCCUPATION_TYPE_Accountants"], ((data["ORGANIZATION_TYPE_Bank"]) * (data["OCCUPATION_TYPE_Accountants"])))) * (
        data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) + (((data["OCCUPATION_TYPE_Accountants"]) * ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.))))))
    v["i264"] = 0.099459 * np.tanh(((((((np.maximum(((data["EXT_SOURCE_3"])), ((((2.0) - (data["EXT_SOURCE_3"])))))) + (
        data["NEW_ANNUITY_TO_INCOME_RATIO"])) / 2.0)) < ((((((2.0) - (data["EXT_SOURCE_3"]))) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.))) * 1.))
    v["i265"] = 0.000780 * np.tanh(((((((((data["ORGANIZATION_TYPE_Mobile"]) - ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > ((((((2.0)) - (
        data["NAME_INCOME_TYPE_Student"]))) - (data["ORGANIZATION_TYPE_Bank"])))) * 1.)))) * 2.0)) - (data["ORGANIZATION_TYPE_Bank"]))) * 2.0))
    v["i266"] = 0.100000 * np.tanh((((-1.0 * ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["ORGANIZATION_TYPE_Mobile"])))))) - (np.maximum(
        (((((data["ORGANIZATION_TYPE_Bank"]) + (data["ORGANIZATION_TYPE_Mobile"])) / 2.0))), (((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (3.0))) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)))))))
    v["i267"] = 0.099899 * np.tanh(((((((data["ORGANIZATION_TYPE_Mobile"]) > (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] > 0, data["FLAG_DOCUMENT_3"], data["ORGANIZATION_TYPE_Mobile"]))) * 1.))
                                     > (np.where(data["ORGANIZATION_TYPE_Mobile"] > 0, data["FLAG_DOCUMENT_3"], (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (((3.0) / 2.0))) / 2.0)))) * 1.))
    v["i268"] = 0.096620 * np.tanh(((np.where(data["NEW_CREDIT_TO_GOODS_RATIO"] < 0, -2.0, ((data["NEW_CREDIT_TO_GOODS_RATIO"]) + (data["NEW_CREDIT_TO_GOODS_RATIO"])))) + (
        np.where(((data["NEW_CREDIT_TO_GOODS_RATIO"]) + (0.318310)) < 0, (-1.0 * ((data["AMT_ANNUITY"]))), data["AMT_ANNUITY"]))))
    v["i269"] = 0.099659 * np.tanh(((np.where(data["NEW_CREDIT_TO_INCOME_RATIO"] < 0, ((data["NAME_INCOME_TYPE_Maternity_leave"]) - (data["AMT_CREDIT"])), data["FLAG_DOCUMENT_3"])) * (
        ((((((((data["FLAG_WORK_PHONE"]) - (data["NEW_CREDIT_TO_INCOME_RATIO"]))) + (data["FLAG_WORK_PHONE"])) / 2.0)) + (data["NAME_CONTRACT_TYPE_Revolving_loans"])) / 2.0))))
    v["i270"] = 0.096999 * np.tanh(((((((((((((data["DAYS_BIRTH"]) * (data["AMT_ANNUITY"]))) * (data["AMT_ANNUITY"]))) * (data["AMT_ANNUITY"]))) - (
        0.318310))) * ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (data["NAME_INCOME_TYPE_Maternity_leave"])) * 1.)))) / 2.0))
    v["i271"] = 0.089701 * np.tanh(np.where((((((data["REGION_POPULATION_RELATIVE"]) * 2.0)) > (1.570796)) * 1.) > 0, -1.0, ((
        (0.636620) + (np.minimum(((((np.tanh((data["REGION_POPULATION_RELATIVE"]))) * 2.0))), ((1.0))))) / 2.0)))
    v["i272"] = 0.100000 * np.tanh((((-1.0) > (((1.570796) * (np.where(((((2.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) - (
        0.636620)) < 0, (-1.0 * ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))), np.tanh((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))))) * 1.))
    v["i273"] = 0.094500 * np.tanh(np.where(data["NAME_EDUCATION_TYPE_Higher_education"] > 0, ((((((data["DAYS_BIRTH"]) < (0.318310)) * 1.)) + (data["AMT_CREDIT"])) / 2.0),
                                            ((data["DAYS_BIRTH"]) * (np.tanh((np.where(data["REGION_RATING_CLIENT"] > 0, data["REGION_RATING_CLIENT"], data["OCCUPATION_TYPE_Managers"])))))))
    v["i274"] = 0.094533 * np.tanh((((-1.0 * ((np.maximum(((data["REFUSED_DAYS_DECISION_MEAN"])), ((data["REGION_RATING_CLIENT_W_CITY"]))))))) * (
        (((((data["REGION_RATING_CLIENT_W_CITY"]) + (data["NAME_CONTRACT_TYPE_Revolving_loans"]))) < (np.tanh((((data["NAME_CONTRACT_TYPE_Revolving_loans"]) * (data["HOUR_APPR_PROCESS_START"])))))) * 1.))))
    v["i275"] = 0.098000 * np.tanh((((data["REFUSED_AMT_GOODS_PRICE_MAX"]) > (np.maximum(((np.where(((data["REFUSED_AMT_ANNUITY_MEAN"]) - (data["REFUSED_APP_CREDIT_PERC_MEAN"]))
                                                                                                    > 0, ((((data["REFUSED_AMT_GOODS_PRICE_MAX"]) * 2.0)) + (data["REFUSED_AMT_ANNUITY_MEAN"])), data["REFUSED_AMT_ANNUITY_MEAN"]))), ((((-1.0) / 2.0)))))) * 1.))
    v["i276"] = 0.096999 * np.tanh(((np.minimum(((data["NAME_CONTRACT_TYPE_Revolving_loans"])), (((-1.0 * ((np.where((((0.636620) + (np.minimum(
        ((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((data["AMT_ANNUITY"]))))) / 2.0) < 0, data["AMT_ANNUITY"], data["NAME_CONTRACT_TYPE_Revolving_loans"])))))))) / 2.0))
    v["i277"] = 0.040300 * np.tanh((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (((((((data["FLAG_EMAIL"]) > (data["AMT_ANNUITY"])) * 1.)) + (
        np.minimum(((((1.570796) * (-2.0)))), ((((data["FLAG_EMAIL"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))))) / 2.0))) * 1.))
    v["i278"] = 0.099200 * np.tanh(((np.where((((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) > (np.tanh((0.636620)))) * 1.) > 0, data["AMT_ANNUITY"], ((
        (((np.tanh((1.570796))) < (np.minimum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((data["FLAG_OWN_REALTY"]))))) * 1.)) / 2.0))) * 2.0))
    v["i279"] = 0.082488 * np.tanh(((np.where(data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"] < -99998, data["AMT_ANNUITY"], (-1.0 * (
        (((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) * ((((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) > ((((-1.0 * ((0.636620)))) / 2.0))) * 1.)) - (data["AMT_ANNUITY"]))))))))) / 2.0))
    v["i280"] = 0.097030 * np.tanh(np.minimum(((((data["REGION_POPULATION_RELATIVE"]) * (((data["EXT_SOURCE_3"]) / 2.0))))), ((((data["AMT_CREDIT"]) * (
        np.where(((data["EXT_SOURCE_3"]) / 2.0) < 0, data["NAME_INCOME_TYPE_Maternity_leave"], ((data["REGION_POPULATION_RELATIVE"]) * (data["EXT_SOURCE_3"])))))))))
    v["i281"] = 0.099501 * np.tanh(np.where(((data["BURO_CREDIT_TYPE_Car_loan_MEAN"]) - ((0.94349050521850586))) < 0, ((np.minimum(
        ((((((0.94349050521850586)) > (data["REGION_RATING_CLIENT"])) * 1.))), ((0.318310)))) / 2.0), ((data["REGION_RATING_CLIENT"]) - (data["BURO_CREDIT_TYPE_Car_loan_MEAN"]))))
    v["i282"] = 0.098896 * np.tanh(((np.where(data["NEW_SOURCES_PROD"] > 0, data["NEW_ANNUITY_TO_INCOME_RATIO"], np.where((((data["NEW_SOURCES_PROD"]) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))
                                                                                                                           * 1.) > 0, data["NEW_SOURCES_PROD"], (((np.maximum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), ((1.570796)))) < (data["HOUR_APPR_PROCESS_START"])) * 1.)))) * 2.0))
    v["i283"] = 0.099500 * np.tanh(np.maximum(((((data["NEW_SCORES_STD"]) * ((((3.141593) < (data["NEW_SCORES_STD"])) * 1.))))), ((
        (((np.tanh((np.tanh((data["NEW_CAR_TO_EMPLOY_RATIO"]))))) > (((np.maximum(((data["AMT_INCOME_TOTAL"])), ((0.318310)))) / 2.0))) * 1.)))))
    v["i284"] = 0.098199 * np.tanh((((((((0.636620) * 2.0)) < (data["OWN_CAR_AGE"])) * 1.)) - (np.where(
        ((data["NEW_CREDIT_TO_INCOME_RATIO"]) - (data["OWN_CAR_AGE"])) < 0, 0.318310, ((((data["NEW_CREDIT_TO_INCOME_RATIO"]) / 2.0)) / 2.0)))))
    v["i285"] = 0.099860 * np.tanh(((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)) * ((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)) > (
        np.maximum((((((data["NEW_INC_PER_CHLD"]) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.))), ((data["AMT_ANNUITY"]))))) * 1.))))
    v["i286"] = 0.079000 * np.tanh((((((np.tanh((1.570796))) < (data["LIVINGAPARTMENTS_MODE"])) * 1.)) * (
        (((((((data["LIVINGAPARTMENTS_AVG"]) / 2.0)) < (1.570796)) * 1.)) * (((((((data["AMT_ANNUITY"]) * 2.0)) * 2.0)) * 2.0))))))
    v["i287"] = 0.099896 * np.tanh(((((((((1.0) + ((((0.636620) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.))) / 2.0)) < (np.tanh(
        ((((np.tanh((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) / 2.0))))) * 1.)) * ((4.11586141586303711))))
    v["i288"] = 0.096495 * np.tanh(np.minimum(((((data["OCCUPATION_TYPE_Laborers"]) * (((((data["AMT_ANNUITY"]) / 2.0)) / 2.0))))), ((np.where(np.maximum(
        ((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((data["AMT_ANNUITY"]))) < 0, 0.0, (((data["OCCUPATION_TYPE_Laborers"]) + (data["NEW_ANNUITY_TO_INCOME_RATIO"])) / 2.0))))))
    v["i289"] = 0.062201 * np.tanh((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (((((np.tanh((np.minimum(((data["OCCUPATION_TYPE_Managers"])), ((((data["NAME_INCOME_TYPE_Working"]) * (
        np.minimum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), (((((0.318310) > (data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.)))))))))))) * 2.0)) * 2.0))) * 1.))
    v["i290"] = 0.097950 * np.tanh(((((((data["AMT_ANNUITY"]) > (data["LIVINGAPARTMENTS_MODE"])) * 1.)) < (np.where(
        ((((((((data["AMT_ANNUITY"]) - (data["AMT_INCOME_TOTAL"]))) * 2.0)) / 2.0)) * 2.0) > 0, data["NEW_INC_PER_CHLD"], data["AMT_INCOME_TOTAL"]))) * 1.))
    v["i291"] = 0.097700 * np.tanh((((((data["NONLIVINGAPARTMENTS_MEDI"]) > (np.where(np.minimum(((data["LIVINGAPARTMENTS_AVG"])), ((data["LIVINGAPARTMENTS_AVG"])))
                                                                                      < 0, 0.318310, (((((data["NONLIVINGAPARTMENTS_AVG"]) > (data["LIVINGAPARTMENTS_AVG"])) * 1.)) + (data["LIVINGAPARTMENTS_MODE"]))))) * 1.)) * 2.0))
    v["i292"] = 0.026250 * np.tanh((((((data["COMMONAREA_MODE"]) > (3.0)) * 1.)) - ((((data["LIVINGAPARTMENTS_AVG"]) > (((data["NONLIVINGAPARTMENTS_AVG"]) - (
        ((((((((data["COMMONAREA_MODE"]) < (data["LIVINGAPARTMENTS_MODE"])) * 1.)) / 2.0)) < (data["NONLIVINGAPARTMENTS_AVG"])) * 1.))))) * 1.))))
    v["i293"] = 0.090030 * np.tanh(((np.where(data["LIVINGAPARTMENTS_MEDI"] > 0, (((((data["LIVINGAPARTMENTS_MEDI"]) < (data["NONLIVINGAPARTMENTS_AVG"])) * 1.))
                                                                                  * (((-2.0) + (data["NONLIVINGAPARTMENTS_MODE"])))), (((data["NONLIVINGAPARTMENTS_AVG"]) > (0.318310)) * 1.))) * 2.0))
    v["i294"] = 0.055511 * np.tanh(np.minimum(((np.where(data["NONLIVINGAPARTMENTS_AVG"] < -99998, np.where(data["LIVINGAPARTMENTS_MODE"] < -99998, (((data["NONLIVINGAPARTMENTS_AVG"]) < (
        data["NONLIVINGAPARTMENTS_MODE"])) * 1.), data["LIVINGAPARTMENTS_MEDI"]), data["NONLIVINGAPARTMENTS_MODE"]))), ((((((((2.32806849479675293)) < (data["NONLIVINGAPARTMENTS_MODE"])) * 1.)) * 2.0)))))
    v["i295"] = 0.039101 * np.tanh(((((((((((((((data["NONLIVINGAPARTMENTS_MODE"]) * ((((((1.570796) + ((0.57018411159515381))))
                                                                                        < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0))
    v["i296"] = 0.073000 * np.tanh((((-1.0 * (((((data["NONLIVINGAPARTMENTS_MEDI"]) > (((((-1.0 * ((np.where(data["LIVINGAPARTMENTS_AVG"] < 0, data["NONLIVINGAPARTMENTS_MEDI"], data[
                                   "NEW_CREDIT_TO_ANNUITY_RATIO"]))))) < ((((((data["NONLIVINGAPARTMENTS_MODE"]) / 2.0)) + (data["LIVINGAPARTMENTS_MODE"])) / 2.0))) * 1.))) * 1.))))) * 2.0))
    v["i297"] = 0.095100 * np.tanh(np.where(data["ORGANIZATION_TYPE_School"] > 0, data["OCCUPATION_TYPE_Laborers"], np.where(data["NAME_TYPE_SUITE_Other_A"] > 0, data["NEW_ANNUITY_TO_INCOME_RATIO"], ((
        data["ORGANIZATION_TYPE_School"]) * (np.where(((data["NEW_ANNUITY_TO_INCOME_RATIO"]) - (data["NAME_TYPE_SUITE_Other_A"])) < 0, data["ORGANIZATION_TYPE_Security_Ministries"], data["OCCUPATION_TYPE_Laborers"]))))))
    v["i298"] = 0.094600 * np.tanh(((-1.0) + ((((((data["AMT_ANNUITY"]) - (1.570796))) < ((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (
        data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.)) + (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) / 2.0)))) / 2.0))) * 1.))))
    v["i299"] = 0.059201 * np.tanh(((((np.where(data["FLAG_DOCUMENT_3"] < 0, data["AMT_ANNUITY"], ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (
        data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) * (data["NAME_TYPE_SUITE_Other_A"])))) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) - (((data["NAME_TYPE_SUITE_Other_A"]) / 2.0))))
    v["i300"] = 0.090199 * np.tanh(((2.0) * ((((((((data["AMT_ANNUITY"]) < (0.636620)) * 1.)) * (((np.where(((((2.06409978866577148)) > (
        data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.) > 0, data["NAME_INCOME_TYPE_Student"], data["AMT_ANNUITY"])) * 2.0)))) * 2.0))))
    v["i301"] = 0.085101 * np.tanh(((np.where(np.minimum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) > 0, data["ORGANIZATION_TYPE_Industry__type_12"], ((
        ((data["ORGANIZATION_TYPE_Industry__type_12"]) * ((-1.0 * ((((data["FLAG_DOCUMENT_3"]) * 2.0))))))) - (data["ORGANIZATION_TYPE_Industry__type_12"])))) - (data["NAME_INCOME_TYPE_Student"])))
    v["i302"] = 0.081773 * np.tanh((((3.0) < (((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] > 0, 3.141593, (((((1.570796) < (
        data["AMT_ANNUITY"])) * 1.)) - (data["ORGANIZATION_TYPE_Industry__type_12"])))) - (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0))))) * 1.))
    v["i303"] = 0.093010 * np.tanh(((((((data["REFUSED_RATE_DOWN_PAYMENT_MAX"]) > (np.tanh((np.tanh((np.tanh((2.0)))))))) * 1.)) < (np.where(data["REFUSED_AMT_DOWN_PAYMENT_MAX"] > 0, np.where(
        data["REFUSED_AMT_DOWN_PAYMENT_MEAN"] > 0, data["REFUSED_AMT_DOWN_PAYMENT_MEAN"], data["REFUSED_RATE_DOWN_PAYMENT_MAX"]), data["REFUSED_AMT_DOWN_PAYMENT_MAX"]))) * 1.))
    v["i304"] = 0.098000 * np.tanh((-1.0 * (((((data["REFUSED_RATE_DOWN_PAYMENT_MAX"]) > (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] > 0, np.where(data["REFUSED_RATE_DOWN_PAYMENT_MAX"]
                                                                                                                                                     > 0, ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0), -1.0), ((data["REFUSED_AMT_DOWN_PAYMENT_MAX"]) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))) * 1.)))))
    v["i305"] = 0.099795 * np.tanh(np.where((((data["AMT_ANNUITY"]) + (data["CC_NAME_CONTRACT_STATUS_Refused_MAX"])) / 2.0) < 0, ((((((data["NEW_CREDIT_TO_INCOME_RATIO"]) + ((((1.0) + (
        data["NAME_INCOME_TYPE_Maternity_leave"])) / 2.0))) / 2.0)) < (data["CC_NAME_CONTRACT_STATUS_Refused_MAX"])) * 1.), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["NEW_CREDIT_TO_INCOME_RATIO"]))))
    v["i306"] = 0.099949 * np.tanh(np.where((((data["CC_AMT_PAYMENT_TOTAL_CURRENT_VAR"]) > ((-1.0 * ((np.maximum(((data["CC_CNT_INSTALMENT_MATURE_CUM_VAR"])), (((((data["CC_AMT_PAYMENT_TOTAL_CURRENT_VAR"]) > (
        ((data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"]) * 2.0))) * 1.))))))))) * 1.) > 0, -2.0, (((data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"]) > (data["CC_AMT_PAYMENT_TOTAL_CURRENT_VAR"])) * 1.)))
    v["i307"] = 0.082296 * np.tanh(((np.minimum((((((((data["CC_AMT_PAYMENT_TOTAL_CURRENT_VAR"]) > (data["CC_NAME_CONTRACT_STATUS_Signed_VAR"])) * 1.)) * (data["CC_NAME_CONTRACT_STATUS_Completed_VAR"])))), ((
        ((((data["CC_NAME_CONTRACT_STATUS_Completed_VAR"]) - (((((data["CC_AMT_PAYMENT_TOTAL_CURRENT_VAR"]) * 2.0)) * 2.0)))) - (data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"])))))) * 2.0))
    v["i308"] = 0.086999 * np.tanh(np.maximum(((data["CC_AMT_DRAWINGS_OTHER_CURRENT_MEAN"])), (((((((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"]) > (np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"] < 0, data[
                                   "CC_CNT_DRAWINGS_POS_CURRENT_MEAN"], ((data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"]) * ((((2.0) > (data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"])) * 1.)))))) * 1.)) * 2.0)) * 2.0)))))
    v["i309"] = 0.099670 * np.tanh((-1.0 * ((np.maximum(((np.where(data["CC_NAME_CONTRACT_STATUS_Signed_VAR"] < 0, np.where(data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"]
                                                                                                                            < -99998, (0.0), 0.318310), np.maximum(((data["CC_NAME_CONTRACT_STATUS_Signed_VAR"])), (((1.0))))))), ((data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"])))))))
    v["i310"] = 0.095000 * np.tanh((((((((((2.0) - (data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"]))) < (np.where(np.where(data["CC_AMT_DRAWINGS_OTHER_CURRENT_MAX"]
                                                                                                                    > 0, data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"], 2.0) > 0, data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"], 3.0))) * 1.)) * 2.0)) * 2.0))
    v["i311"] = 0.098999 * np.tanh(((3.0) * (np.minimum(((((((3.0) * (((data["NAME_INCOME_TYPE_Student"]) * (data["NEW_DOC_IND_KURT"]))))) * 2.0))), ((
        ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (data["NAME_CONTRACT_TYPE_Revolving_loans"])) * 1.)))))))))
    v["i312"] = 0.098189 * np.tanh(((np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"] > 0, (((data["CC_CNT_DRAWINGS_POS_CURRENT_MEAN"]) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) / 2.0), (((np.where(
        data["NEW_CREDIT_TO_ANNUITY_RATIO"] > 0, (8.0), (((data["CC_CNT_DRAWINGS_POS_CURRENT_MEAN"]) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) / 2.0))) < (data["CC_CNT_DRAWINGS_OTHER_CURRENT_MEAN"])) * 1.))) * 2.0))
    v["i313"] = 0.099849 * np.tanh((-1.0 * (((((data["CC_NAME_CONTRACT_STATUS_Completed_VAR"]) > (np.where(data["CC_NAME_CONTRACT_STATUS_Active_VAR"] < 0, (-1.0 * (
        ((((((data["CC_AMT_PAYMENT_TOTAL_CURRENT_VAR"]) * 2.0)) > (data["CC_NAME_CONTRACT_STATUS_Completed_VAR"])) * 1.)))), ((0.636620) + (2.0))))) * 1.)))))
    v["i314"] = 0.099020 * np.tanh(np.where(data["CC_AMT_DRAWINGS_POS_CURRENT_MEAN"] < -99998, 0.0, ((((((data["CC_CNT_DRAWINGS_POS_CURRENT_MEAN"]) - (
        data["CC_AMT_DRAWINGS_POS_CURRENT_MEAN"]))) - (data["CC_AMT_DRAWINGS_POS_CURRENT_MEAN"]))) - (((data["CC_AMT_DRAWINGS_POS_CURRENT_MEAN"]) / 2.0)))))
    v["i315"] = 0.098960 * np.tanh(np.minimum(((((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)) - (data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"]))) - ((((3.141593) < (
        data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.)))) - (data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"])))), (((((3.0) < (data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.)))))
    v["i316"] = 0.066799 * np.tanh((((((np.maximum(((data["CC_AMT_DRAWINGS_POS_CURRENT_MAX"])), ((1.570796)))) > (np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"] > 0, np.maximum(
        ((data["CC_AMT_DRAWINGS_POS_CURRENT_MAX"])), ((np.where(data["CC_AMT_DRAWINGS_POS_CURRENT_MEAN"] < 0, data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"], data["CC_AMT_DRAWINGS_POS_CURRENT_MEAN"])))), 2.0))) * 1.)) / 2.0))
    v["i317"] = 0.099501 * np.tanh((-1.0 * (((((((data["CC_CNT_INSTALMENT_MATURE_CUM_VAR"]) > (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, (4.0), (-1.0 * (
        ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.))) * 1.))))))) * 1.)) * ((6.73373746871948242)))))))
    v["i318"] = 0.098100 * np.tanh(((((((np.where(data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"] < 0, data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"], 0.318310)) + (data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"])) / 2.0)) > (
        (((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) > (np.tanh((np.maximum(((data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"])), ((data["CC_CNT_DRAWINGS_OTHER_CURRENT_MAX"]))))))) * 1.))) * 1.))
    v["i319"] = 0.095460 * np.tanh((((((3.0)) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) * ((((((11.21043491363525391)) + (
        (11.21043491363525391)))) * ((-1.0 * (((((((1.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) > (3.141593)) * 1.)))))))))
    v["i320"] = 0.089963 * np.tanh(((np.minimum(((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)) * (((data["NEW_CREDIT_TO_INCOME_RATIO"]) * (
        ((data["NEW_ANNUITY_TO_INCOME_RATIO"]) / 2.0)))))) - (data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"])))), (((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (2.0)) * 1.))))) * 2.0))
    v["i321"] = 0.084251 * np.tanh(((((data["ORGANIZATION_TYPE_Industry__type_12"]) + (((((np.maximum((((((3.141593) < (data["NEW_CREDIT_TO_INCOME_RATIO"])) * 1.))),
                                                                                                      ((data["ORGANIZATION_TYPE_Industry__type_11"])))) * 2.0)) * 2.0)))) * ((-1.0 * ((data["AMT_ANNUITY"]))))))
    v["i322"] = 0.070000 * np.tanh((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (((np.minimum((((((np.minimum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])),
                                                                                                           ((((3.141593) / 2.0))))) < (data["AMT_ANNUITY"])) * 1.))), ((((0.318310) * (data["AMT_ANNUITY"])))))) - (-2.0)))) * 1.))
    v["i323"] = 0.060300 * np.tanh(((((((data["NAME_INCOME_TYPE_Maternity_leave"]) + ((((((data["POS_NAME_CONTRACT_STATUS_Signed_MEAN"]) - (
        ((-2.0) * (np.where(data["NEW_CREDIT_TO_INCOME_RATIO"] < 0, data["POS_SK_DPD_MEAN"], (2.30259108543395996))))))) > ((6.0))) * 1.)))) * 2.0)) * 2.0))
    v["i324"] = 0.060646 * np.tanh(np.where(((((data["POS_SK_DPD_MEAN"]) - (np.tanh((data["POS_SK_DPD_MAX"]))))) * (data["POS_SK_DPD_MEAN"])) < 0, ((
        data["POS_SK_DPD_MAX"]) * (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) - (data["POS_SK_DPD_MAX"])))), data["NEW_ANNUITY_TO_INCOME_RATIO"]))
    v["i325"] = 0.098599 * np.tanh((((((data["DAYS_BIRTH"]) > (np.maximum(((((data["DAYS_BIRTH"]) * 2.0))), ((data["ORGANIZATION_TYPE_University"]))))) * 1.)) - (
        ((((((np.maximum(((((data["ORGANIZATION_TYPE_Industry__type_12"]) / 2.0))), ((data["ORGANIZATION_TYPE_University"])))) / 2.0)) / 2.0)) / 2.0))))
    v["i326"] = 0.070002 * np.tanh(np.where((((data["ORGANIZATION_TYPE_Industry__type_1"]) < (data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.) > 0, data["ORGANIZATION_TYPE_Industry__type_1"], np.where(
        data["NEW_ANNUITY_TO_INCOME_RATIO"] > 0, data["NAME_INCOME_TYPE_Working"], ((((((np.tanh((1.0))) + (data["NAME_INCOME_TYPE_Working"])) / 2.0)) < (data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.))))
    v["i327"] = 0.091899 * np.tanh(((((((((((((((8.0)) < (data["NEW_CREDIT_TO_INCOME_RATIO"])) * 1.)) + (data["ORGANIZATION_TYPE_Industry__type_3"])) / 2.0)) * (
        (-1.0 * ((data["NEW_CREDIT_TO_INCOME_RATIO"])))))) + (data["ORGANIZATION_TYPE_Industry__type_3"])) / 2.0)) * ((-1.0 * ((data["NEW_ANNUITY_TO_INCOME_RATIO"]))))))
    v["i328"] = 0.097300 * np.tanh(np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] < 0, np.maximum(((data["ORGANIZATION_TYPE_Legal_Services"])), ((((data["OCCUPATION_TYPE_Low_skill_Laborers"]) * (
        data["NAME_INCOME_TYPE_Working"]))))), (-1.0 * (((((((((data["NAME_INCOME_TYPE_Working"]) * (data["OCCUPATION_TYPE_Low_skill_Laborers"]))) * 2.0)) > (data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.))))))
    v["i329"] = 0.099970 * np.tanh(((((((10.0)) < (np.maximum(((((data["AMT_ANNUITY"]) * (data["NEW_CREDIT_TO_INCOME_RATIO"])))), ((
        (((((9.67386054992675781)) - (((data["AMT_ANNUITY"]) * (data["NEW_CREDIT_TO_INCOME_RATIO"]))))) - (1.0))))))) * 1.)) * 2.0))
    v["i330"] = 0.100000 * np.tanh(np.minimum(((((((((data["NEW_CREDIT_TO_INCOME_RATIO"]) > (0.636620)) * 1.)) > (((1.570796) + (
        (-1.0 * ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))))) * 1.))), ((((1.570796) + ((-1.0 * ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))))))
    v["i331"] = 0.099985 * np.tanh(((data["NAME_CONTRACT_TYPE_Cash_loans"]) * (np.where(np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, data["ORGANIZATION_TYPE_Trade__type_4"], data["AMT_INCOME_TOTAL"])
                                                                                        > 0, (-1.0 * ((((data["AMT_INCOME_TOTAL"]) + (data["NAME_CONTRACT_TYPE_Revolving_loans"]))))), ((data["NAME_CONTRACT_TYPE_Revolving_loans"]) * (data["ORGANIZATION_TYPE_Trade__type_4"]))))))
    v["i332"] = 0.099999 * np.tanh(((((((data["NEW_INC_PER_CHLD"]) * ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < (np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] < 0, data[
                                   "OCCUPATION_TYPE_Low_skill_Laborers"], data["EXT_SOURCE_1"]))) * 1.)))) * (2.0))) * ((((-1.0) < (data["EXT_SOURCE_1"])) * 1.))))
    v["i333"] = 0.099500 * np.tanh(((((((6.92336940765380859)) < (((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)) * 2.0))) * 1.)) - (
        ((((((data["OCCUPATION_TYPE_Low_skill_Laborers"]) + ((((data["OCCUPATION_TYPE_Low_skill_Laborers"]) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0))) / 2.0))) / 2.0)) > (1.0)) * 1.))))
    v["i334"] = 0.064600 * np.tanh((-1.0 * (((((((data["AMT_CREDIT"]) - ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (np.minimum(((np.where(
        data["NEW_ANNUITY_TO_INCOME_RATIO"] > 0, data["CNT_CHILDREN"], data["NEW_CREDIT_TO_ANNUITY_RATIO"]))), ((data["AMT_ANNUITY"]))))) * 1.)))) > (np.tanh((1.570796)))) * 1.)))))
    v["i335"] = 0.092697 * np.tanh(((((np.where(data["NAME_FAMILY_STATUS_Married"] > 0, (((data["NAME_INCOME_TYPE_Student"]) > (data["NEW_CREDIT_TO_INCOME_RATIO"])) * 1.), np.where(
        data["NEW_CREDIT_TO_INCOME_RATIO"] > 0, data["NEW_CREDIT_TO_INCOME_RATIO"], np.where(data["AMT_ANNUITY"] > 0, data["NAME_INCOME_TYPE_Maternity_leave"], data["NAME_FAMILY_STATUS_Married"])))) / 2.0)) / 2.0))
    v["i336"] = 0.090054 * np.tanh(np.where(data["AMT_ANNUITY"] < 0, np.where(np.where((((data["AMT_ANNUITY"]) < (data["NAME_FAMILY_STATUS_Single___not_married"])) * 1.) > 0, data[
                                   "NAME_FAMILY_STATUS_Single___not_married"], ((data["CNT_CHILDREN"]) / 2.0)) < 0, data["NAME_INCOME_TYPE_Maternity_leave"], data["CNT_CHILDREN"]), ((data["NAME_FAMILY_STATUS_Single___not_married"]) / 2.0)))
    v["i337"] = 0.085501 * np.tanh(((np.minimum(((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) / 2.0))), ((((np.where(
        (((data["FLAG_EMP_PHONE"]) < (data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"])) * 1.) > 0, data["CODE_GENDER"], data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * (data["CODE_GENDER"])))))) / 2.0))
    v["i338"] = 0.010999 * np.tanh(np.where(data["NAME_CONTRACT_TYPE_Revolving_loans"] > 0, np.where(data["ORGANIZATION_TYPE_Trade__type_4"] > 0, data["ORGANIZATION_TYPE_Trade__type_4"], (((-1.0 * (
        (data["ORGANIZATION_TYPE_Kindergarten"])))) + ((-1.0 * (((((data["ORGANIZATION_TYPE_Kindergarten"]) < (data["AMT_ANNUITY"])) * 1.))))))), (-1.0 * ((data["ORGANIZATION_TYPE_Trade__type_4"])))))
    v["i339"] = 0.096004 * np.tanh(((((np.where(((data["NAME_INCOME_TYPE_Maternity_leave"]) * (data["WEEKDAY_APPR_PROCESS_START_MONDAY"])) > 0, np.minimum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((
        (-1.0 * ((data["NAME_INCOME_TYPE_Maternity_leave"])))))), ((data["NAME_INCOME_TYPE_Student"]) - (data["NEW_ANNUITY_TO_INCOME_RATIO"])))) * (data["ORGANIZATION_TYPE_Kindergarten"]))) / 2.0))
    v["i340"] = 0.095596 * np.tanh(np.where((((data["AMT_GOODS_PRICE"]) > ((0.41270980238914490))) * 1.) > 0, data["NEW_CREDIT_TO_GOODS_RATIO"], np.where(
        data["AMT_GOODS_PRICE"] > 0, data["AMT_GOODS_PRICE"], ((np.maximum(((np.tanh((1.0)))), ((data["NEW_CREDIT_TO_GOODS_RATIO"])))) - (1.0)))))
    v["i341"] = 0.089984 * np.tanh(((0.636620) * ((-1.0 * ((np.where(data["NAME_CONTRACT_TYPE_Revolving_loans"] > 0, data["AMT_ANNUITY"], ((
        data["AMT_ANNUITY"]) * (np.where(data["NEW_DOC_IND_KURT"] > 0, (((1.0) < (data["NAME_CONTRACT_TYPE_Revolving_loans"])) * 1.), -2.0))))))))))
    v["i342"] = 0.074030 * np.tanh((((((((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, np.where(data["NAME_INCOME_TYPE_Pensioner"] < 0, 1.0, data[
                                   "NEW_CREDIT_TO_ANNUITY_RATIO"]), data["AMT_ANNUITY"])) < (-1.0)) * 1.)) * (((-1.0) - (data["DAYS_BIRTH"]))))) * 2.0))
    v["i343"] = 0.089964 * np.tanh(((data["ORGANIZATION_TYPE_Trade__type_2"]) * (((((1.0)) + (np.where(data["ORGANIZATION_TYPE_Trade__type_2"] < 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], np.minimum(
        ((((data["FLAG_PHONE"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))), ((((data["ORGANIZATION_TYPE_Trade__type_2"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))))) / 2.0))))
    v["i344"] = 0.000099 * np.tanh((-1.0 * ((((np.where(data["FLAG_PHONE"] < 0, data["ORGANIZATION_TYPE_Trade__type_2"], data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * (np.where(
        (((1.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) / 2.0) > 0, data["NEW_ANNUITY_TO_INCOME_RATIO"], (((data["FLAG_PHONE"]) + (data["NEW_ANNUITY_TO_INCOME_RATIO"])) / 2.0))))))))
    v["i345"] = 0.095300 * np.tanh(np.where(data["FLAG_PHONE"] > 0, np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] > 0, (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (
        (-1.0 * ((-2.0))))) * 1.), (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (((data["ORGANIZATION_TYPE_Trade__type_2"]) * 2.0))) * 1.)), data["ORGANIZATION_TYPE_Trade__type_2"]))
    v["i346"] = 0.003000 * np.tanh(((((np.where(((data["NEW_DOC_IND_KURT"]) + (((data["NEW_DOC_IND_KURT"]) * (data["AMT_ANNUITY"])))) > 0, data[
                                   "NEW_DOC_IND_KURT"], np.where(data["AMT_ANNUITY"] > 0, data["NEW_ANNUITY_TO_INCOME_RATIO"], data["AMT_ANNUITY"]))) / 2.0)) / 2.0))
    v["i347"] = 0.089001 * np.tanh(((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (((((data["AMT_ANNUITY"]) * (data["AMT_ANNUITY"]))) * (
        (((2.0) < (((data["AMT_ANNUITY"]) * ((-1.0 * ((((data["NEW_DOC_IND_KURT"]) + (data["NAME_CONTRACT_TYPE_Cash_loans"]))))))))) * 1.))))))
    v["i348"] = 0.072951 * np.tanh(((np.where(data["FLAG_DOCUMENT_3"] < 0, (((((((data["NAME_CONTRACT_TYPE_Revolving_loans"]) + (data["NEW_DOC_IND_KURT"]))) / 2.0))
                                                                             < (data["FLAG_DOCUMENT_3"])) * 1.), ((data["NAME_CONTRACT_TYPE_Revolving_loans"]) + (data["NEW_DOC_IND_KURT"])))) * 2.0))
    v["i349"] = 0.098823 * np.tanh(np.where(data["NAME_CONTRACT_TYPE_Revolving_loans"] > 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], (-1.0 * (((((data["NAME_INCOME_TYPE_Student"]) > (
        (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (np.maximum(((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["NEW_ANNUITY_TO_INCOME_RATIO"])))), ((np.tanh((1.570796))))))) / 2.0))) * 1.))))))
    v["i350"] = 0.020000 * np.tanh((((((((((np.where(((data["NAME_INCOME_TYPE_Maternity_leave"]) + (data["NEW_ANNUITY_TO_INCOME_RATIO"])) < 0, data[
                                   "NEW_ANNUITY_TO_INCOME_RATIO"], data["NAME_INCOME_TYPE_Student"])) > (data["NAME_INCOME_TYPE_Student"])) * 1.)) * 2.0)) * 2.0)) - (data["NAME_INCOME_TYPE_Student"])))
    v["i351"] = 0.098004 * np.tanh(np.where(data["AMT_ANNUITY"] > 0, ((np.where(((((data["ORGANIZATION_TYPE_Trade__type_3"]) / 2.0)) + (
        data["AMT_ANNUITY"])) < 0, (14.66671943664550781), data["ORGANIZATION_TYPE_Mobile"])) * (-1.0)), data["ORGANIZATION_TYPE_Mobile"]))
    v["i352"] = 0.099584 * np.tanh(np.where(data["ORGANIZATION_TYPE_Trade__type_3"] < 0, (((((0.636620) < (data["LIVINGAPARTMENTS_AVG"])) * 1.)) * (np.where(
        data["NEW_CREDIT_TO_ANNUITY_RATIO"] > 0, data["LIVINGAPARTMENTS_AVG"], -1.0))), ((data["LIVINGAPARTMENTS_AVG"]) - (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)))))
    v["i353"] = 0.000020 * np.tanh(((((((((((data["LIVINGAPARTMENTS_MEDI"]) > (2.0)) * 1.)) < ((((-2.0) < (data["LIVINGAPARTMENTS_MEDI"])) * 1.)))
                                        * 1.)) * (((data["NONLIVINGAPARTMENTS_AVG"]) * (data["NEW_ANNUITY_TO_INCOME_RATIO"]))))) * 2.0))
    v["i354"] = 0.065970 * np.tanh(((data["NONLIVINGAPARTMENTS_AVG"]) * (((np.where(data["NONLIVINGAPARTMENTS_MODE"] < 0, data["AMT_ANNUITY"], (((np.maximum(
        ((data["NONLIVINGAPARTMENTS_MODE"])), ((data["LIVINGAPARTMENTS_MODE"])))) > (2.0)) * 1.))) * ((((data["LIVINGAPARTMENTS_MODE"]) > (-1.0)) * 1.))))))
    v["i355"] = 0.086500 * np.tanh(((((np.minimum((((((-1.0) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.))), ((((np.maximum(
        ((data["LIVINGAPARTMENTS_AVG"])), (((((((0.636620) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) < (data["AMT_ANNUITY"])) * 1.))))) * 2.0))))) * 2.0)) * 2.0))
    v["i356"] = 0.098650 * np.tanh(np.minimum(((np.where((((1.570796) + (data["AMT_ANNUITY"])) / 2.0) < 0, (((3.0) + (data["NONLIVINGAPARTMENTS_MEDI"])) / 2.0),
                                                         (((data["LIVINGAPARTMENTS_MODE"]) > ((3.0))) * 1.)))), (((((data["NONLIVINGAPARTMENTS_MEDI"]) > (data["AMT_ANNUITY"])) * 1.)))))
    v["i357"] = 0.018997 * np.tanh(np.minimum((((-1.0 * ((((data["NONLIVINGAPARTMENTS_AVG"]) * ((((data["LIVINGAPARTMENTS_MEDI"]) > (0.636620)) * 1.)))))))), ((
        np.where(((data["NONLIVINGAPARTMENTS_AVG"]) + (((data["LIVINGAPARTMENTS_MODE"]) / 2.0))) < -99998, 3.0, data["NONLIVINGAPARTMENTS_AVG"])))))
    v["i358"] = 0.042000 * np.tanh((-1.0 * ((((((((data["LIVINGAPARTMENTS_MEDI"]) > (data["AMT_ANNUITY"])) * 1.)) < (np.where(data["NONLIVINGAPARTMENTS_AVG"]
                                                                                                                              < 0, (((data["LIVINGAPARTMENTS_MEDI"]) + ((0.22441512346267700))) / 2.0), (-1.0 * ((((data["AMT_ANNUITY"]) / 2.0))))))) * 1.)))))
    v["i359"] = 0.097998 * np.tanh(np.where(data["INSTAL_PAYMENT_DIFF_VAR"] < -99998, (((-1.0 * ((data["AMT_ANNUITY"])))) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"])), ((
        data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * ((((data["INSTAL_PAYMENT_DIFF_VAR"]) > (((((data["INSTAL_PAYMENT_DIFF_VAR"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) / 2.0))) * 1.)))))
    v["i360"] = 0.070050 * np.tanh(np.where(((((data["INSTAL_PAYMENT_DIFF_VAR"]) - (((data["INSTAL_DBD_STD"]) + (0.318310))))) + (
        data["INSTAL_PAYMENT_DIFF_VAR"])) > 0, np.where(data["INSTAL_DBD_STD"] < 0, data["INSTAL_DBD_STD"], (10.57927513122558594)), 0.318310))
    v["i361"] = 0.096000 * np.tanh(np.minimum(((((data["CC_AMT_PAYMENT_TOTAL_CURRENT_VAR"]) - (data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"])))), ((((((np.where(data["CC_NAME_CONTRACT_STATUS_Active_VAR"] > 0, data[
                                   "CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"], (14.66551780700683594))) - (data["CC_AMT_PAYMENT_TOTAL_CURRENT_VAR"]))) * ((((data["CC_AMT_PAYMENT_TOTAL_CURRENT_VAR"]) > (2.0)) * 1.)))))))
    v["i362"] = 0.099265 * np.tanh(((((np.maximum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), ((data["ORGANIZATION_TYPE_Industry__type_1"])))) * (((data["ORGANIZATION_TYPE_Industry__type_1"]) * (
        np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, (11.86987972259521484), np.minimum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))))))) * 2.0))
    v["i363"] = 0.095700 * np.tanh(np.where(data["AMT_ANNUITY"] < 0, np.where(np.maximum(((data["NEW_LIVE_IND_SUM"])), ((data["NAME_INCOME_TYPE_Maternity_leave"]))) < 0, np.where(
        data["NEW_ANNUITY_TO_INCOME_RATIO"] < 0, data["NAME_INCOME_TYPE_Maternity_leave"], data["FLAG_EMAIL"]), ((data["NEW_ANNUITY_TO_INCOME_RATIO"]) / 2.0)), (-1.0 * ((data["ORGANIZATION_TYPE_Trade__type_4"])))))
    v["i364"] = 0.096810 * np.tanh(((np.where((-1.0 * ((data["AMT_ANNUITY"]))) < 0, (-1.0 * (((((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]) > (1.570796)) * 1.)))), ((
        (((((2.0)) < (data["INSTAL_DAYS_ENTRY_PAYMENT_STD"])) * 1.)) < (data["INSTAL_PAYMENT_PERC_STD"])) * 1.))) * (data["INSTAL_DAYS_ENTRY_PAYMENT_STD"])))
    v["i365"] = 0.100000 * np.tanh(((((1.0) - (np.maximum(((np.minimum(((data["FONDKAPREMONT_MODE_reg_oper_spec_account"])), ((data["LIVINGAPARTMENTS_MODE"]))))), ((
        (((np.where(data["AMT_ANNUITY"] < 0, data["LIVINGAPARTMENTS_MODE"], ((data["LIVINGAPARTMENTS_MODE"]) / 2.0))) < ((3.0))) * 1.))))))) * 2.0))
    v["i366"] = 0.018860 * np.tanh((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > ((((np.maximum((((((10.0)) + (data["LIVINGAPARTMENTS_AVG"])))),
                                                                                            ((np.tanh((np.where(data["AMT_ANNUITY"] > 0, np.tanh(((10.0))), data["AMT_ANNUITY"]))))))) + ((4.0))) / 2.0))) * 1.))
    v["i367"] = 0.099050 * np.tanh((-1.0 * ((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, data["NAME_EDUCATION_TYPE_Academic_degree"], ((
        ((((((-1.0 * ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) / 2.0)) - (data["NAME_INCOME_TYPE_Student"]))) < ((((-2.0) + (data["INSTAL_PAYMENT_DIFF_VAR"])) / 2.0))) * 1.))))))
    v["i368"] = 0.095009 * np.tanh(((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] > 0, data["FONDKAPREMONT_MODE_org_spec_account"], (((data["OCCUPATION_TYPE_Cleaning_staff"]) < (
        ((data["OCCUPATION_TYPE_Cleaning_staff"]) * ((((3.141593) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) / 2.0))))) * 1.))) * (((data["OCCUPATION_TYPE_Cleaning_staff"]) - (data["AMT_ANNUITY"])))))
    v["i369"] = 0.082589 * np.tanh((-1.0 * ((((3.141593) * ((((3.141593) < (np.where(data["OCCUPATION_TYPE_Cleaning_staff"] > 0, ((
        3.141593) * ((((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]) + (3.0)) / 2.0))), data["INSTAL_PAYMENT_DIFF_VAR"]))) * 1.)))))))
    v["i370"] = 0.081997 * np.tanh(np.where(data["INSTAL_DAYS_ENTRY_PAYMENT_STD"] > 0, data["ORGANIZATION_TYPE_Transport__type_3"], ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (
        ((data["ORGANIZATION_TYPE_Transport__type_3"]) * (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (np.maximum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))))))))))
    v["i371"] = 0.091909 * np.tanh(np.where((((data["NEW_CREDIT_TO_INCOME_RATIO"]) > (data["AMT_ANNUITY"])) * 1.) > 0, data["NAME_INCOME_TYPE_Maternity_leave"], ((
        ((8.14943599700927734)) < (((data["NEW_CREDIT_TO_INCOME_RATIO"]) - (((np.tanh((data["AMT_ANNUITY"]))) - ((8.27048969268798828))))))) * 1.)))
    v["i372"] = 0.090395 * np.tanh(((data["NAME_TYPE_SUITE_Children"]) * (((data["NAME_TYPE_SUITE_Children"]) * ((((np.tanh((data["AMT_ANNUITY"]))) + (
        ((((1.47119557857513428)) + (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] > 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], data["NEW_ANNUITY_TO_INCOME_RATIO"]))) / 2.0))) / 2.0))))))
    v["i373"] = 0.092002 * np.tanh(np.minimum((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (2.0)) * 1.))), (((((((1.0) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)) - (np.where(
        (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (data["NAME_TYPE_SUITE_Spouse__partner"])) * 1.) > 0, data["NEW_ANNUITY_TO_INCOME_RATIO"], data["ORGANIZATION_TYPE_Trade__type_4"])))))))
    v["i374"] = 0.089650 * np.tanh((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (2.0)) * 1.)) * ((-1.0 * (
        (((((((((np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"] > 0, ((data["AMT_ANNUITY"]) * 2.0), data["NAME_TYPE_SUITE_Spouse__partner"])) * 2.0)) * 2.0)) * 2.0)) * 2.0)))))))
    v["i375"] = 0.099860 * np.tanh(((((np.where(data["NAME_TYPE_SUITE_Spouse__partner"] > 0, 2.0, data["NEW_CREDIT_TO_INCOME_RATIO"])) - (data["NEW_CREDIT_TO_INCOME_RATIO"]))) * (
        (((((((data["ORGANIZATION_TYPE_Government"]) < (data["NEW_CREDIT_TO_INCOME_RATIO"])) * 1.)) + (data["NEW_CREDIT_TO_INCOME_RATIO"]))) - (data["ORGANIZATION_TYPE_Government"])))))
    v["i376"] = 0.028342 * np.tanh((-1.0 * ((((((np.where(np.where(data["NEW_CREDIT_TO_INCOME_RATIO"] < 0, data["NAME_TYPE_SUITE_Spouse__partner"],
                                                                   np.minimum(((data["FLAG_EMAIL"])), ((data["NEW_ANNUITY_TO_INCOME_RATIO"])))) < 0, 0.0, data["FLAG_EMAIL"])) * 2.0)) * 2.0)))))
    v["i377"] = 0.049000 * np.tanh((((-1.0 * (((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > ((((((2.0) > (data["AMT_CREDIT"])) * 1.)) + (
        (((np.minimum(((3.0)), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) < (data["AMT_CREDIT"])) * 1.))))) * 1.))))) / 2.0))
    v["i378"] = 0.097398 * np.tanh((((((np.where((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + ((((((data["NAME_TYPE_SUITE_Other_B"]) * (data["NAME_TYPE_SUITE_Other_B"])))
                                                                                              < (data["EXT_SOURCE_1"])) * 1.))) / 2.0) < 0, data["NAME_INCOME_TYPE_Student"], 1.570796)) < (data["EXT_SOURCE_1"])) * 1.)) * (-2.0)))
    v["i379"] = 0.099776 * np.tanh((((-1.0 * ((data["OCCUPATION_TYPE_Laborers"])))) * ((((((((((np.where(data["EXT_SOURCE_1"] < -99998, (1.77269029617309570),
                                                                                                         (((data["OCCUPATION_TYPE_Laborers"]) < (2.0)) * 1.))) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)) * 2.0)) * 2.0)) * 2.0))))
    v["i380"] = 0.087500 * np.tanh((-1.0 * (((((data["NAME_INCOME_TYPE_Student"]) + ((((np.where(data["AMT_ANNUITY"] < 0, data["NAME_INCOME_TYPE_Student"], np.maximum(
        ((np.maximum(((data["AMT_ANNUITY"])), ((data["NAME_INCOME_TYPE_Student"]))))), ((data["OCCUPATION_TYPE_Laborers"]))))) < (data["AMT_CREDIT"])) * 1.))) / 2.0)))))
    v["i381"] = 0.061010 * np.tanh((-1.0 * ((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["ORGANIZATION_TYPE_Trade__type_7"]))) * (np.where(data["AMT_ANNUITY"]
                                                                                                                                                > 0, np.where(data["ORGANIZATION_TYPE_Trade__type_7"] > 0, data["DAYS_BIRTH"], data["NAME_TYPE_SUITE_Other_B"]), data["ORGANIZATION_TYPE_Trade__type_7"])))))))
    v["i382"] = 0.094702 * np.tanh(np.where(data["DAYS_BIRTH"] > 0, np.where(data["OCCUPATION_TYPE_Drivers"] > 0, data["ORGANIZATION_TYPE_Trade__type_7"],
                                                                             (-1.0 * ((data["ORGANIZATION_TYPE_Trade__type_7"])))), np.tanh((np.tanh(((((data["DAYS_BIRTH"]) < ((-1.0 * ((1.570796))))) * 1.)))))))
    v["i383"] = 0.099500 * np.tanh(np.where((((data["AMT_ANNUITY"]) < ((1.65035283565521240))) * 1.) > 0, np.where(data["ORGANIZATION_TYPE_Trade__type_7"] > 0, ((
        (data["NEW_ANNUITY_TO_INCOME_RATIO"]) < (data["AMT_ANNUITY"])) * 1.), ((((1.55915415287017822)) < (data["AMT_ANNUITY"])) * 1.)), data["FLAG_DOCUMENT_3"]))
    v["i384"] = 0.072983 * np.tanh((((12.58773136138916016)) * (np.where(data["DAYS_REGISTRATION"] > 0, data[
                                   "NAME_INCOME_TYPE_Maternity_leave"], (((((((((7.83792781829833984)) + (-2.0)) / 2.0)) < (data["AMT_ANNUITY"])) * 1.)) * 2.0)))))
    v["i385"] = 0.062950 * np.tanh(((((((((np.minimum(((-1.0)), ((((data["AMT_ANNUITY"]) * 2.0))))) - (data["DAYS_ID_PUBLISH"]))) < (
        ((((((0.318310) < (data["AMT_ANNUITY"])) * 1.)) < (data["AMT_ANNUITY"])) * 1.))) * 1.)) + (-1.0)) / 2.0))
    v["i386"] = 0.090988 * np.tanh(((np.minimum(((np.where(data["LIVINGAPARTMENTS_MEDI"] < -99998, (((data["DAYS_BIRTH"]) < (-1.0)) * 1.), np.where(
        data["NEW_ANNUITY_TO_INCOME_RATIO"] < 0, data["DAYS_BIRTH"], data["LIVINGAPARTMENTS_MEDI"])))), (((((data["DAYS_BIRTH"]) < (0.636620)) * 1.))))) / 2.0))
    v["i387"] = 0.099989 * np.tanh(np.minimum((((((data["FLOORSMIN_MEDI"]) < ((((np.where(data["LIVINGAREA_MODE"] > 0, data["ELEVATORS_MODE"], data["LIVINGAREA_MODE"])) > (
        data["ELEVATORS_MEDI"])) * 1.))) * 1.))), (((((np.minimum(((data["FLOORSMIN_MEDI"])), ((data["ELEVATORS_MEDI"])))) > (data["ENTRANCES_MODE"])) * 1.)))))
    v["i388"] = 0.099710 * np.tanh(np.maximum(((np.tanh((data["NEW_CAR_TO_EMPLOY_RATIO"])))), ((np.minimum((((((((data["NAME_INCOME_TYPE_Pensioner"]) + (
        data["EXT_SOURCE_1"])) / 2.0)) * (((((data["NAME_INCOME_TYPE_Pensioner"]) * 2.0)) * 2.0))))), ((np.maximum(((0.0)), ((data["NAME_INCOME_TYPE_Pensioner"]))))))))))
    v["i389"] = 0.080000 * np.tanh(np.minimum((((((data["NEW_EMPLOY_TO_BIRTH_RATIO"]) < (data["DAYS_EMPLOYED"])) * 1.))), ((((np.where(((data["NEW_EMPLOY_TO_BIRTH_RATIO"]) - (
        data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"])) < 0, data["NAME_INCOME_TYPE_Pensioner"], 3.141593)) - (((data["NEW_EMPLOY_TO_BIRTH_RATIO"]) - (data["NAME_INCOME_TYPE_Pensioner"]))))))))
    v["i390"] = 0.100000 * np.tanh((-1.0 * ((((np.where(((((((1.0) + (1.570796)) / 2.0)) + (data["NEW_SOURCES_PROD"])) / 2.0)
                                                        > 0, data["DAYS_BIRTH"], (((-1.0 * ((0.318310)))) * (data["DAYS_BIRTH"])))) * 2.0)))))
    v["i391"] = 0.084999 * np.tanh(((((-1.0 * ((data["DAYS_BIRTH"])))) + (((((((data["DAYS_EMPLOYED"]) > (0.636620)) * 1.)) > (
        ((((-1.0 * ((data["DAYS_BIRTH"])))) + ((((0.636620) > (data["NEW_EMPLOY_TO_BIRTH_RATIO"])) * 1.))) / 2.0))) * 1.))) / 2.0))
    v["i392"] = 0.090389 * np.tanh(np.where(data["NEW_SOURCES_PROD"] > 0, ((data["EXT_SOURCE_3"]) - (data["NEW_SOURCES_PROD"])), ((
        (data["NEW_SOURCES_PROD"]) > (np.tanh((((((data["CLOSED_AMT_CREDIT_SUM_MEAN"]) - (1.0))) - (data["NEW_SOURCES_PROD"])))))) * 1.)))
    v["i393"] = 0.099520 * np.tanh((((np.minimum(((((((2.0)) > (data["CLOSED_AMT_CREDIT_SUM_MEAN"])) * 1.))), ((np.where(data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"] < 0, data[
                                   "CLOSED_AMT_CREDIT_SUM_MEAN"], data["CLOSED_AMT_CREDIT_SUM_SUM"]))))) > ((((((data["CLOSED_AMT_CREDIT_SUM_SUM"]) > (np.tanh((data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"])))) * 1.)) / 2.0))) * 1.))
    v["i394"] = 0.075001 * np.tanh(((((((((((data["NEW_CAR_TO_BIRTH_RATIO"]) > (data["CODE_GENDER"])) * 1.)) > (data["CODE_GENDER"])) * 1.)) * (
        ((np.where(data["NEW_CAR_TO_BIRTH_RATIO"] < -99998, 0.0, data["CODE_GENDER"])) - (data["NEW_EMPLOY_TO_BIRTH_RATIO"]))))) / 2.0))
    v["i395"] = 0.099623 * np.tanh(np.where((((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) > (3.141593)) * 1.) > 0, (-1.0 * ((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]))),
                                            ((np.tanh((np.tanh((np.tanh((((data["EXT_SOURCE_3"]) * ((-1.0 * ((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])))))))))))) / 2.0)))
    v["i396"] = 0.099000 * np.tanh(np.where(data["BURO_CREDIT_TYPE_Mortgage_MEAN"] > 0, -2.0, ((np.where(data["BURO_CREDIT_TYPE_Consumer_credit_MEAN"] < 0, (((data["BURO_CREDIT_TYPE_Consumer_credit_MEAN"]) < (
        data["AMT_REQ_CREDIT_BUREAU_DAY"])) * 1.), ((data["AMT_REQ_CREDIT_BUREAU_DAY"]) - ((((data["BURO_CREDIT_TYPE_Consumer_credit_MEAN"]) < (data["AMT_REQ_CREDIT_BUREAU_YEAR"])) * 1.))))) / 2.0)))
    v["i397"] = 0.099702 * np.tanh(np.where(data["AMT_REQ_CREDIT_BUREAU_MON"] > 0, ((data["BURO_CREDIT_TYPE_Mortgage_MEAN"]) * (((data["BURO_CREDIT_TYPE_Another_type_of_loan_MEAN"]) + (data["AMT_REQ_CREDIT_BUREAU_MON"])))),
                                            (-1.0 * ((np.maximum((((((data["BURO_CREDIT_TYPE_Mortgage_MEAN"]) < (data["AMT_REQ_CREDIT_BUREAU_MON"])) * 1.))), ((((data["BURO_CREDIT_TYPE_Mortgage_MEAN"]) * 2.0)))))))))
    v["i398"] = 0.095995 * np.tanh(np.minimum(((np.where(((data["BURO_CREDIT_TYPE_Mortgage_MEAN"]) - (data["AMT_REQ_CREDIT_BUREAU_MON"])) > 0, (-1.0 * (
        (data["BURO_CREDIT_TYPE_Mortgage_MEAN"]))), (((6.0)) - (data["AMT_REQ_CREDIT_BUREAU_MON"]))))), (((((1.570796) < (data["AMT_REQ_CREDIT_BUREAU_QRT"])) * 1.)))))
    v["i399"] = 0.086499 * np.tanh((((((-1.0 * ((np.where((((0.318310) < (((data["BURO_CREDIT_TYPE_Car_loan_MEAN"]) - (3.141593)))) * 1.) > 0, data[
                                   "BURO_CREDIT_TYPE_Car_loan_MEAN"], (-1.0 * (((((2.0) < (data["BURO_CREDIT_TYPE_Car_loan_MEAN"])) * 1.))))))))) * 2.0)) * 2.0))
    v["i400"] = 0.047958 * np.tanh(np.where(data["BURO_CNT_CREDIT_PROLONG_SUM"] < 0, np.minimum((((((((data["BURO_CREDIT_TYPE_Another_type_of_loan_MEAN"]) * (data["CNT_CHILDREN"]))) < (
        data["AMT_REQ_CREDIT_BUREAU_WEEK"])) * 1.))), ((((data["AMT_REQ_CREDIT_BUREAU_WEEK"]) * (data["BURO_CREDIT_TYPE_Another_type_of_loan_MEAN"]))))), ((data["CNT_CHILDREN"]) + (-2.0))))
    v["i401"] = 0.089300 * np.tanh((((((data["EXT_SOURCE_1"]) > (np.where(data["EXT_SOURCE_1"] > 0, (((data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]) < (data["AMT_ANNUITY"])) * 1.), data[
                                   "AMT_ANNUITY"]))) * 1.)) * ((((((data["NAME_INCOME_TYPE_Student"]) > (data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"])) * 1.)) + (data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"])))))
    v["i402"] = 0.098002 * np.tanh(((np.where((-1.0 * ((data["AMT_REQ_CREDIT_BUREAU_WEEK"]))) > 0, (((3.141593) < (np.where(data["AMT_REQ_CREDIT_BUREAU_YEAR"]
                                                                                                                            < 0, data["AMT_REQ_CREDIT_BUREAU_QRT"], data["AMT_REQ_CREDIT_BUREAU_DAY"]))) * 1.), data["AMT_REQ_CREDIT_BUREAU_QRT"])) * 2.0))
    v["i403"] = 0.056801 * np.tanh((-1.0 * (((((np.maximum(((data["BURO_CREDIT_TYPE_Another_type_of_loan_MEAN"])), ((
        data["BURO_CREDIT_ACTIVE_Closed_MEAN"])))) > (np.where(data["AMT_REQ_CREDIT_BUREAU_QRT"] < 0, (13.24810218811035156), 0.318310))) * 1.)))))
    v["i404"] = 0.091995 * np.tanh(((((((((data["BURO_CREDIT_TYPE_Credit_card_MEAN"]) + ((0.24336224794387817)))) > (data["AMT_REQ_CREDIT_BUREAU_YEAR"])) * 1.)) < (
        ((((((data["BURO_CREDIT_TYPE_Credit_card_MEAN"]) > (data["AMT_REQ_CREDIT_BUREAU_HOUR"])) * 1.)) > ((((2.0) < (data["AMT_REQ_CREDIT_BUREAU_QRT"])) * 1.))) * 1.))) * 1.))
    v["i405"] = 0.013991 * np.tanh(np.where(data["AMT_REQ_CREDIT_BUREAU_WEEK"] < -99998, 0.318310, ((np.maximum(((((data["BURO_CREDIT_TYPE_Another_type_of_loan_MEAN"]) - (
        data["AMT_REQ_CREDIT_BUREAU_DAY"])))), ((data["AMT_REQ_CREDIT_BUREAU_MON"])))) * (((data["AMT_REQ_CREDIT_BUREAU_WEEK"]) - (data["AMT_REQ_CREDIT_BUREAU_DAY"]))))))
    v["i406"] = 0.018490 * np.tanh((((-1.0 * (((3.0))))) * (((((((3.0)) < ((((((np.maximum(((data["AMT_REQ_CREDIT_BUREAU_YEAR"])), (((1.03554749488830566)))))
                                                                               * 2.0)) + ((((data["AMT_REQ_CREDIT_BUREAU_MON"]) + (data["BURO_CNT_CREDIT_PROLONG_SUM"])) / 2.0))) / 2.0))) * 1.)) * 2.0))))
    v["i407"] = 0.080770 * np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"] > 0, np.minimum(((np.minimum(((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])), ((data["AMT_REQ_CREDIT_BUREAU_MON"]))))),
                                                                                                     ((data["AMT_REQ_CREDIT_BUREAU_YEAR"]))), (((2.67786097526550293)) * ((-1.0 * (((((data["AMT_REQ_CREDIT_BUREAU_HOUR"]) < (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])) * 1.))))))))
    v["i408"] = 0.086000 * np.tanh((((((6.0)) + (data["AMT_REQ_CREDIT_BUREAU_DAY"]))) * ((-1.0 * (((((np.where(data["EXT_SOURCE_1"] < -99998, ((
        data["AMT_REQ_CREDIT_BUREAU_WEEK"]) - ((-1.0 * ((data["AMT_REQ_CREDIT_BUREAU_MON"]))))), data["EXT_SOURCE_1"])) > ((1.48802196979522705))) * 1.)))))))
    v["i409"] = 0.092999 * np.tanh(np.maximum((((-1.0 * ((((1.0) / 2.0)))))), ((np.where((((data["EXT_SOURCE_1"]) + (
        2.0)) / 2.0) > 0, (-1.0 * ((((data["EXT_SOURCE_1"]) - ((-1.0 * ((data["DAYS_BIRTH"])))))))), data["DAYS_BIRTH"])))))
    v["i410"] = 0.099581 * np.tanh(np.where(((data["ORGANIZATION_TYPE_XNA"]) * (-1.0)) > 0, ((np.where(data["DAYS_BIRTH"]
                                                                                                       > 0, 0.318310, data["DAYS_BIRTH"])) * (data["CODE_GENDER"])), (((((-1.0 * ((data["DAYS_BIRTH"])))) / 2.0)) / 2.0)))
    v["i411"] = 0.097800 * np.tanh(np.where(((data["NEW_SOURCES_PROD"]) + (data["EXT_SOURCE_1"])) < 0, np.where(data["EXT_SOURCE_1"] < -99998, 0.0, ((
        (data["DAYS_BIRTH"]) < (np.tanh(((((data["EXT_SOURCE_1"]) < (data["NAME_INCOME_TYPE_Student"])) * 1.))))) * 1.)), data["AMT_ANNUITY"]))
    v["i412"] = 0.030003 * np.tanh(np.where((((data["LIVINGAREA_MODE"]) + (np.where(data["LIVINGAREA_MEDI"] > 0, -2.0, data["LIVINGAREA_AVG"]))) / 2.0) > 0, data[
                                   "NEW_ANNUITY_TO_INCOME_RATIO"], np.where(data["LIVINGAREA_MODE"] > 0, data["NAME_INCOME_TYPE_Student"], (-1.0 * ((((0.318310) / 2.0)))))))
    v["i413"] = 0.089859 * np.tanh((((-1.0 * ((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] > 0, np.where(((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (
        data["CODE_GENDER"])) > 0, 0.636620, data["DAYS_BIRTH"]), ((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["CODE_GENDER"]))))))) / 2.0))
    v["i414"] = 0.099998 * np.tanh(((np.where(data["NEW_CAR_TO_EMPLOY_RATIO"] > 0, 0.318310, (-1.0 * (((((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) / 2.0)) < (
        data["NEW_CAR_TO_EMPLOY_RATIO"])) * 1.)))))) - (np.where((-1.0 * ((data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"]))) < 0, data["AMT_ANNUITY"], data["NAME_INCOME_TYPE_Student"]))))
    v["i415"] = 0.100000 * np.tanh(((data["WALLSMATERIAL_MODE_Stone__brick"]) * (np.where((((2.0) + (data["NEW_ANNUITY_TO_INCOME_RATIO"])) / 2.0) > 0, ((
        (data["LIVINGAREA_MODE"]) > (((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (data["NEW_ANNUITY_TO_INCOME_RATIO"]))) + (2.0)))) * 1.), 3.141593))))
    v["i416"] = 0.010500 * np.tanh(((((data["AMT_ANNUITY"]) * ((((13.51322269439697266)) * (np.where(np.where(data["LIVINGAREA_MEDI"] < -99998, data["AMT_ANNUITY"], data[
                                   "NEW_ANNUITY_TO_INCOME_RATIO"]) > 0, data["NAME_INCOME_TYPE_Student"], (((data["LIVINGAREA_MEDI"]) > ((7.19499874114990234))) * 1.))))))) * 2.0))
    v["i417"] = 0.097300 * np.tanh(((np.where(np.where(data["FONDKAPREMONT_MODE_reg_oper_account"] < 0, (7.0), data["AMT_ANNUITY"]) < 0, ((
        (data["LIVINGAREA_MEDI"]) > (data["HOUSETYPE_MODE_block_of_flats"])) * 1.), ((((7.0)) < (data["LIVINGAREA_MEDI"])) * 1.))) * ((7.0))))
    v["i418"] = 0.050100 * np.tanh(((((data["ACTIVE_AMT_ANNUITY_MAX"]) + (np.minimum(((((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]) + (data["ACTIVE_AMT_ANNUITY_MAX"])))), ((
        data["CLOSED_MONTHS_BALANCE_MAX_MAX"])))))) * ((((data["NAME_INCOME_TYPE_Student"]) < (((((data["ACTIVE_AMT_ANNUITY_MAX"]) / 2.0)) / 2.0))) * 1.))))
    v["i419"] = 0.089991 * np.tanh(np.where(np.where(((data["ORGANIZATION_TYPE_Security"]) * (data["AMT_ANNUITY"])) > 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], data[
                                   "OCCUPATION_TYPE_Security_staff"]) < 0, ((data["ORGANIZATION_TYPE_Security"]) * (((data["AMT_ANNUITY"]) * (data["AMT_ANNUITY"])))), (-1.0 * ((data["ORGANIZATION_TYPE_Security"])))))
    v["i420"] = 0.090001 * np.tanh(np.where((((((data["AMT_INCOME_TOTAL"]) + (2.0))) > (1.570796)) * 1.) > 0, (((3.141593) < (
        data["AMT_ANNUITY"])) * 1.), ((((1.570796) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))
    v["i421"] = 0.099915 * np.tanh((((((np.where(((data["ORGANIZATION_TYPE_XNA"]) / 2.0) < -99998, data["DAYS_EMPLOYED"], data["AMT_INCOME_TOTAL"])) > (
        (((data["DAYS_EMPLOYED"]) > ((-1.0 * ((((((6.68204927444458008)) > (data["DAYS_EMPLOYED"])) * 1.)))))) * 1.))) * 1.)) * (data["DAYS_EMPLOYED"])))
    v["i422"] = 0.059958 * np.tanh((-1.0 * ((((data["ORGANIZATION_TYPE_Transport__type_2"]) - ((((2.88201642036437988)) * (((((np.where(data["AMT_ANNUITY"] > 0, data[
                                   "ORGANIZATION_TYPE_Transport__type_2"], data["NEW_INC_PER_CHLD"])) * (data["ORGANIZATION_TYPE_Transport__type_2"]))) * (data["AMT_ANNUITY"]))))))))))
    v["i423"] = 0.083100 * np.tanh((-1.0 * ((np.where(data["HOUSETYPE_MODE_block_of_flats"] > 0, (((data["REG_REGION_NOT_LIVE_REGION"]) + ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["ORGANIZATION_TYPE_Telecom"])) / 2.0))
                                                                                                   ) / 2.0), ((np.where(data["REG_REGION_NOT_LIVE_REGION"] < -99998, data["REG_REGION_NOT_LIVE_REGION"], data["NAME_TYPE_SUITE_Other_A"])) - (data["ORGANIZATION_TYPE_Telecom"])))))))
    v["i424"] = 0.038796 * np.tanh(((((((0.85919755697250366)) < (np.where((((0.85919755697250366)) - (data["AMT_CREDIT"])) < 0, np.where(
        data["AMT_CREDIT"] < -99998, 2.0, data["NEW_CREDIT_TO_ANNUITY_RATIO"]), ((data["AMT_CREDIT"]) - (data["NEW_ANNUITY_TO_INCOME_RATIO"]))))) * 1.)) / 2.0))
    v["i425"] = 0.094983 * np.tanh(np.minimum((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, ((data["FLAG_OWN_CAR"]) / 2.0), ((
        data["OCCUPATION_TYPE_Cooking_staff"]) * (data["AMT_INCOME_TOTAL"]))))) * 1.))), ((((data["DAYS_BIRTH"]) * (((data["OCCUPATION_TYPE_Cooking_staff"]) / 2.0)))))))
    v["i426"] = 0.064559 * np.tanh(np.where(data["ORGANIZATION_TYPE_Industry__type_11"] < 0, (((((1.0) / 2.0)) < (np.where(((data["OCCUPATION_TYPE_Cooking_staff"]) + (
        data["AMT_ANNUITY"])) < 0, ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["AMT_ANNUITY"])), data["ORGANIZATION_TYPE_Transport__type_2"]))) * 1.), data["NEW_CREDIT_TO_ANNUITY_RATIO"]))
    v["i427"] = 0.099554 * np.tanh(np.where(np.where(data["REG_CITY_NOT_WORK_CITY"] < 0, (((data["ORGANIZATION_TYPE_Industry__type_11"]) < (data["AMT_CREDIT"])) * 1.), data["AMT_CREDIT"])
                                            < 0, (((data["ORGANIZATION_TYPE_Industry__type_11"]) + (np.tanh((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) / 2.0), (((2.0) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)))
    v["i428"] = 0.097400 * np.tanh(((np.minimum(((np.maximum(((data["POS_SK_DPD_MAX"])), ((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (
        data["POS_SK_DPD_MAX"]))) - (data["POS_SK_DPD_MAX"]))))))), ((((((data["POS_SK_DPD_MEAN"]) - (data["POS_SK_DPD_MAX"]))) * 2.0))))) * 2.0))
    v["i429"] = 0.082300 * np.tanh(((np.where(data["POS_SK_DPD_MAX"] < 0, data["POS_SK_DPD_MAX"], ((3.0) * 2.0))) * (
        (((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) / 2.0)) > (np.where(data["AMT_ANNUITY"] > 0, 0.318310, 1.0))) * 1.)) * 2.0))))
    v["i430"] = 0.087798 * np.tanh((-1.0 * ((((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < -99998, 1.570796, np.where(
        (((((np.tanh(((2.21421408653259277)))) * 2.0)) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.) > 0, (7.0), data["NAME_INCOME_TYPE_Student"]))) * ((10.0)))))))
    v["i431"] = 0.062602 * np.tanh((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (np.where(data["NAME_EDUCATION_TYPE_Secondary___secondary_special"] > 0, ((((13.09417152404785156)) + (
        ((((((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]) - (data["NAME_INCOME_TYPE_State_servant"]))) * 2.0)) * 2.0))) / 2.0), 1.570796))) * 1.)) * 2.0))
    v["i432"] = 0.099650 * np.tanh((((((data["AMT_INCOME_TOTAL"]) < (np.minimum(((((((np.tanh((data["ORGANIZATION_TYPE_Telecom"]))) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) + (
        data["NEW_CREDIT_TO_ANNUITY_RATIO"])))), (((((0.318310) < (data["AMT_INCOME_TOTAL"])) * 1.)))))) * 1.)) * (data["OCCUPATION_TYPE_High_skill_tech_staff"])))
    v["i433"] = 0.099800 * np.tanh((((((-1.0 * (((13.76046848297119141))))) * ((((data["AMT_CREDIT"]) > (3.0)) * 1.))))
                                    * (((data["LIVINGAPARTMENTS_MEDI"]) + ((((-1.0 * ((1.0)))) + (data["AMT_ANNUITY"])))))))
    v["i434"] = 0.096440 * np.tanh((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > ((((3.141593) + (np.maximum(
        ((((np.tanh((1.570796))) / 2.0))), (((((((6.62170648574829102)) / 2.0)) * (data["AMT_ANNUITY"]))))))) / 2.0))) * 1.)) * 2.0)) * 2.0))
    v["i435"] = 0.069395 * np.tanh(np.minimum(((((((((np.tanh((data["AMT_INCOME_TOTAL"]))) > (data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.)) > ((((2.0) < (
        data["AMT_ANNUITY"])) * 1.))) * 1.))), ((((((((data["NAME_INCOME_TYPE_Student"]) > (data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.)) < (data["AMT_ANNUITY"])) * 1.)))))
    v["i436"] = 0.096980 * np.tanh((-1.0 * ((np.minimum(((np.maximum(((data["ORGANIZATION_TYPE_Trade__type_4"])), ((((data["NEW_CAR_TO_BIRTH_RATIO"]) - (
        data["NEW_ANNUITY_TO_INCOME_RATIO"]))))))), ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (((data["NEW_CAR_TO_BIRTH_RATIO"]) - (data["FLAG_OWN_CAR"])))))))))))
    v["i437"] = 0.097900 * np.tanh((((((((data["NEW_CAR_TO_EMPLOY_RATIO"]) + (data["OWN_CAR_AGE"]))) > (np.where(
        data["NEW_CAR_TO_EMPLOY_RATIO"] > 0, (((0.318310) + (data["LIVE_CITY_NOT_WORK_CITY"])) / 2.0), ((data["OWN_CAR_AGE"]) * 2.0)))) * 1.)) * 2.0))
    v["i438"] = 0.065499 * np.tanh(((((data["OWN_CAR_AGE"]) * ((((data["AMT_INCOME_TOTAL"]) > (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (
        ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0))))) * 1.)))) * ((((data["AMT_INCOME_TOTAL"]) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.))))
    v["i439"] = 0.087788 * np.tanh(((((((((((-1.0) + (((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (-2.0)) / 2.0)) + (-2.0)) / 2.0))) / 2.0)) > (
        np.where(data["NEW_CAR_TO_BIRTH_RATIO"] > 0, data["NEW_ANNUITY_TO_INCOME_RATIO"], data["LIVE_CITY_NOT_WORK_CITY"]))) * 1.)) * 2.0)) * 2.0))
    v["i440"] = 0.099998 * np.tanh(np.where(((1.570796) + (data["NEW_ANNUITY_TO_INCOME_RATIO"])) < 0, data["NEW_CAR_TO_BIRTH_RATIO"], np.minimum(
        (((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < (data["FLAG_OWN_CAR"])) * 1.))), ((((((data["NAME_INCOME_TYPE_Student"]) * ((11.31053447723388672)))) * (data["FLAG_OWN_CAR"])))))))
    v["i441"] = 0.085490 * np.tanh(((np.where(data["LIVINGAPARTMENTS_AVG"] < 0, ((data["ORGANIZATION_TYPE_Government"]) * (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, data[
                                   "ORGANIZATION_TYPE_Government"], data["NAME_INCOME_TYPE_Student"]))), ((data["WEEKDAY_APPR_PROCESS_START_MONDAY"]) - (data["ORGANIZATION_TYPE_Government"])))) * (((data["AMT_INCOME_TOTAL"]) * 2.0))))
    v["i442"] = 0.099900 * np.tanh(((0.318310) * ((-1.0 * ((np.where(((-1.0) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) < 0, data["NAME_INCOME_TYPE_Student"],
                                                                     np.where((((data["AMT_ANNUITY"]) + (data["LIVINGAPARTMENTS_MODE"])) / 2.0) < -99998, 1.570796, data["WEEKDAY_APPR_PROCESS_START_MONDAY"]))))))))
    v["i443"] = 0.099970 * np.tanh(np.where(data["NONLIVINGAPARTMENTS_MEDI"] > 0, data["LIVE_CITY_NOT_WORK_CITY"], ((((((data["WALLSMATERIAL_MODE_Wooden"]) > (data["NONLIVINGAPARTMENTS_MEDI"])) * 1.)) < (
        np.tanh(((((np.where(data["LIVINGAPARTMENTS_MEDI"] > 0, data["LIVE_CITY_NOT_WORK_CITY"], data["NONLIVINGAPARTMENTS_MEDI"])) + (data["NEW_ANNUITY_TO_INCOME_RATIO"])) / 2.0))))) * 1.)))
    v["i444"] = 0.097640 * np.tanh(np.where(data["NONLIVINGAPARTMENTS_AVG"] < -99998, 0.0, ((((data["AMT_ANNUITY"]) * (((((data["AMT_ANNUITY"]) * (
        data["NONLIVINGAPARTMENTS_AVG"]))) * (data["AMT_ANNUITY"]))))) * ((((data["AMT_ANNUITY"]) + (data["NONLIVINGAPARTMENTS_MEDI"])) / 2.0)))))
    v["i445"] = 0.070959 * np.tanh(((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < ((1.0))) * 1.)) < (np.where(data["CLOSED_MONTHS_BALANCE_MIN_MIN"] > 0, (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (
        data["LIVE_CITY_NOT_WORK_CITY"])) * 1.), ((data["CLOSED_MONTHS_BALANCE_MAX_MAX"]) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["LIVE_CITY_NOT_WORK_CITY"]))))))) * 1.))
    v["i446"] = 0.078109 * np.tanh(np.where(np.where(data["BURO_STATUS_C_MEAN_MEAN"] < 0, data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"], ((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]) * (
        data["BURO_STATUS_2_MEAN_MEAN"]))) > 0, data["CLOSED_MONTHS_BALANCE_MIN_MIN"], np.tanh(((((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]) > (data["CLOSED_MONTHS_BALANCE_MIN_MIN"])) * 1.)))))
    v["i447"] = 0.050001 * np.tanh(np.minimum((((((data["BURO_STATUS_0_MEAN_MEAN"]) + ((-1.0 * ((data["CLOSED_MONTHS_BALANCE_MIN_MIN"]))))) / 2.0))), ((
        ((((-1.0 * ((data["CLOSED_MONTHS_BALANCE_MIN_MIN"])))) < (((data["BURO_STATUS_0_MEAN_MEAN"]) + (((data["CLOSED_MONTHS_BALANCE_MIN_MIN"]) * (data["BURO_STATUS_0_MEAN_MEAN"])))))) * 1.)))))
    v["i448"] = 0.099952 * np.tanh(np.where(data["CLOSED_MONTHS_BALANCE_MIN_MIN"] > 0, ((data["BURO_STATUS_4_MEAN_MEAN"]) - (data["BURO_MONTHS_BALANCE_MAX_MAX"])), ((
        data["BURO_STATUS_0_MEAN_MEAN"]) * (np.where(data["BURO_MONTHS_BALANCE_MAX_MAX"] < 0, ((data["BURO_STATUS_4_MEAN_MEAN"]) - (data["BURO_STATUS_0_MEAN_MEAN"])), data["BURO_STATUS_4_MEAN_MEAN"])))))
    v["i449"] = 0.075350 * np.tanh((-1.0 * (((((((((-1.0 * ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]))))))
                                                  < (data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"])) * 1.)) < (np.tanh((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"])))) * 1.)))))
    v["i450"] = 0.098650 * np.tanh(((np.where(data["AMT_ANNUITY"] < -99998, data["NAME_INCOME_TYPE_Student"], (((1.570796) < (
        ((((((2.0) + (np.where(data["AMT_ANNUITY"] < 0, data["CLOSED_MONTHS_BALANCE_MIN_MIN"], data["AMT_INCOME_TOTAL"]))) / 2.0)) + (data["CLOSED_MONTHS_BALANCE_MIN_MIN"])) / 2.0))) * 1.))) * 2.0))
    v["i451"] = 0.066996 * np.tanh(((((((np.where(np.where(data["BURO_STATUS_2_MEAN_MEAN"] < 0, (-1.0 * ((data["BURO_STATUS_2_MEAN_MEAN"]))), ((-2.0) + (
        data["BURO_STATUS_2_MEAN_MEAN"]))) < 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], (-1.0 * ((data["NAME_INCOME_TYPE_Student"]))))) * 2.0)) * 2.0)) * 2.0))
    v["i452"] = 0.094900 * np.tanh(((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]) * ((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (np.maximum(
        (((((((np.tanh((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) < ((2.93883275985717773))) * 1.))), ((data["AMT_ANNUITY"])))))) > ((2.93883275985717773))) * 1.))))
    v["i453"] = 0.095479 * np.tanh(((((((((((((((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]) * (data["BURO_STATUS_C_MEAN_MEAN"]))) * 2.0)) + (
        data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"])) / 2.0)) < (((data["BURO_STATUS_4_MEAN_MEAN"]) - (3.141593)))) * 1.)) * 2.0)) * 2.0))
    v["i454"] = 0.030401 * np.tanh((-1.0 * (((((np.where(data["BURO_STATUS_2_MEAN_MEAN"] > 0, ((-2.0) / 2.0), np.where((((1.570796) < (
        data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"])) * 1.) > 0, ((-2.0) / 2.0), (1.74125826358795166)))) < (data["CLOSED_MONTHS_BALANCE_MIN_MIN"])) * 1.)))))
    v["i455"] = 0.032510 * np.tanh((-1.0 * (((((((((1.87832641601562500)) < (data["BURO_STATUS_2_MEAN_MEAN"])) * 1.)) < (np.where(data["BURO_STATUS_2_MEAN_MEAN"]
                                                                                                                                  > 0, data["BURO_MONTHS_BALANCE_MAX_MAX"], (((data["CLOSED_MONTHS_BALANCE_MIN_MIN"]) > ((((2.0) + (1.570796)) / 2.0))) * 1.)))) * 1.)))))
    v["i456"] = 0.096860 * np.tanh(((data["AMT_ANNUITY"]) * ((((((((data["CLOSED_MONTHS_BALANCE_MAX_MAX"]) + (1.570796))) < (data["BURO_MONTHS_BALANCE_MAX_MAX"])) * 1.)) + (
        np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < -99998, 2.0, (((1.570796) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)))))))
    v["i457"] = 0.099244 * np.tanh((-1.0 * (((((np.where((((data["BURO_STATUS_0_MEAN_MEAN"]) + (np.tanh((data["BURO_STATUS_2_MEAN_MEAN"])))) / 2.0) > 0, np.where(
        data["NEW_CREDIT_TO_INCOME_RATIO"] > 0, data["BURO_STATUS_0_MEAN_MEAN"], data["BURO_STATUS_2_MEAN_MEAN"]), 3.0)) < (data["NEW_CREDIT_TO_INCOME_RATIO"])) * 1.)))))
    v["i458"] = 0.096120 * np.tanh((((((((((0.636620) > (((3.141593) - (data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"])))) * 1.)) * 2.0)) - (
        np.tanh(((((((2.0) - (data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]))) < (0.636620)) * 1.)))))) * 2.0))
    v["i459"] = 0.095590 * np.tanh(np.maximum(((np.maximum(((data["ORGANIZATION_TYPE_Telecom"])), (((((((data["AMT_ANNUITY"]) > (3.141593)) * 1.)) * (
        data["AMT_ANNUITY"]))))))), ((np.where(data["AMT_ANNUITY"] > 0, data["ORGANIZATION_TYPE_Transport__type_1"], data["ORGANIZATION_TYPE_Telecom"])))))
    v["i460"] = 0.099833 * np.tanh(np.minimum((((((((((((data["NAME_INCOME_TYPE_Student"]) + ((-1.0 * ((data["NEW_ANNUITY_TO_INCOME_RATIO"]))))) / 2.0)) * (data["ORGANIZATION_TYPE_Construction"]))) - (
        data["NAME_EDUCATION_TYPE_Academic_degree"]))) - (data["ORGANIZATION_TYPE_Transport__type_1"])))), ((data["ORGANIZATION_TYPE_Transport__type_1"]))))
    v["i461"] = 0.044999 * np.tanh(((((((((((np.minimum((((-1.0 * ((data["OCCUPATION_TYPE_Secretaries"]))))), ((
        ((data["NAME_TYPE_SUITE_Other_B"]) * ((-1.0 * ((data["NAME_TYPE_SUITE_Other_B"]))))))))) * (data["AMT_ANNUITY"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0))
    v["i462"] = 0.036997 * np.tanh(np.where(data["NAME_EDUCATION_TYPE_Lower_secondary"] > 0, data["OCCUPATION_TYPE_Cooking_staff"], (((-1.0 * (((((data["OCCUPATION_TYPE_Cooking_staff"]) < (np.where(
        data["NEW_ANNUITY_TO_INCOME_RATIO"] > 0, data["OCCUPATION_TYPE_Cooking_staff"], np.minimum(((data["ORGANIZATION_TYPE_Industry__type_12"])), ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * 2.0))))))) * 1.))))) * 2.0)))
    v["i463"] = 0.097979 * np.tanh((-1.0 * ((np.where(np.where(data["AMT_ANNUITY"] < 0, (((data["OCCUPATION_TYPE_Cooking_staff"]) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.), data["OCCUPATION_TYPE_Cooking_staff"])
                                                      > 0, np.where(data["AMT_INCOME_TOTAL"] > 0, data["OCCUPATION_TYPE_Cooking_staff"], (-1.0 * ((data["NAME_INCOME_TYPE_Maternity_leave"])))), data["ORGANIZATION_TYPE_Industry__type_12"])))))
    v["i464"] = 0.097701 * np.tanh(((data["WALLSMATERIAL_MODE_Wooden"]) * (((np.where(data["WALLSMATERIAL_MODE_Wooden"] > 0, np.where(data["AMT_ANNUITY"] > 0, data[
                                   "NEW_CREDIT_TO_ANNUITY_RATIO"], data["AMT_ANNUITY"]), data["WALLSMATERIAL_MODE_Mixed"])) * ((((data["AMT_ANNUITY"]) + ((-1.0 * ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))) / 2.0))))))
    v["i465"] = 0.087593 * np.tanh(((data["FONDKAPREMONT_MODE_not_specified"]) * (((data["FONDKAPREMONT_MODE_not_specified"]) * (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (np.where(
        np.where(data["AMT_INCOME_TOTAL"] > 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], data["NEW_ANNUITY_TO_INCOME_RATIO"]) > 0, data["NEW_ANNUITY_TO_INCOME_RATIO"], data["AMT_INCOME_TOTAL"]))))))))
    v["i466"] = 0.019960 * np.tanh(np.minimum((((((data["WALLSMATERIAL_MODE_Wooden"]) > (((((data["AMT_ANNUITY"]) / 2.0)) / 2.0))) * 1.))), ((np.where(
        data["WEEKDAY_APPR_PROCESS_START_SATURDAY"] < 0, ((((-1.0 * ((data["OCCUPATION_TYPE_Drivers"])))) < (data["OCCUPATION_TYPE_Drivers"])) * 1.), (-1.0 * ((data["OCCUPATION_TYPE_Drivers"]))))))))
    v["i467"] = 0.046600 * np.tanh((((((((3.141593) + ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (((data["AMT_ANNUITY"]) + (data["OCCUPATION_TYPE_Drivers"])))) * 1.))))
                                       < ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["AMT_ANNUITY"])))) / 2.0))) * 1.)) * 2.0))
    v["i468"] = 0.067994 * np.tanh((-1.0 * ((np.where(data["WALLSMATERIAL_MODE_Wooden"] < 0, (((((0.318310) + (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * 2.0))) / 2.0)) * (
        data["ORGANIZATION_TYPE_Culture"])), (((data["NAME_INCOME_TYPE_Student"]) < ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (0.318310)) / 2.0))) * 1.))))))
    v["i469"] = 0.077995 * np.tanh((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (data["AMT_INCOME_TOTAL"])) * 1.)) * (((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (
        data["AMT_INCOME_TOTAL"])) * 1.)) < (data["AMT_INCOME_TOTAL"])) * 1.)) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))) * (data["NEW_ANNUITY_TO_INCOME_RATIO"])))
    v["i470"] = 0.099770 * np.tanh(((((data["ORGANIZATION_TYPE_Culture"]) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (np.minimum(((np.where(((data["ORGANIZATION_TYPE_Culture"]) + (data["NEW_ANNUITY_TO_INCOME_RATIO"]))
                                                                                                                                             > 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], data["OCCUPATION_TYPE_Drivers"]))), ((data["NAME_EDUCATION_TYPE_Lower_secondary"])))))))) * (data["NAME_EDUCATION_TYPE_Lower_secondary"])))
    v["i471"] = 0.091999 * np.tanh((-1.0 * ((((((np.where(((data["AMT_INCOME_TOTAL"]) + (((0.318310) + (data["NAME_INCOME_TYPE_Student"])))) < 0, data[
                                   "NAME_EDUCATION_TYPE_Lower_secondary"], np.where(data["AMT_INCOME_TOTAL"] < 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], data["NAME_EDUCATION_TYPE_Lower_secondary"]))) / 2.0)) / 2.0)))))
    v["i472"] = 0.083551 * np.tanh(((data["OCCUPATION_TYPE_Drivers"]) * (np.where(((data["CLOSED_MONTHS_BALANCE_MIN_MIN"]) + (np.minimum(((data["AMT_ANNUITY"])), ((
        (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (data["CLOSED_MONTHS_BALANCE_MIN_MIN"])) * 1.)))))) > 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], (((-1.0) < (data["CLOSED_MONTHS_BALANCE_MIN_MIN"])) * 1.)))))
    v["i473"] = 0.087699 * np.tanh((((6.01758050918579102)) * ((((6.01758050918579102)) * ((((6.01758050918579102)) * (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (
        (((data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]) > (((((6.01758050918579102)) > (data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.))) * 1.))))))))))
    v["i474"] = 0.099500 * np.tanh((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, ((
        ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (2.0))) / 2.0), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (2.0))))) * 1.)) * (data["CODE_GENDER"])))
    v["i475"] = 0.097000 * np.tanh((((((np.where(0.636620 > 0, (((((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]) * (data["AMT_INCOME_TOTAL"]))) + (
        ((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]) * (((data["AMT_INCOME_TOTAL"]) * 2.0))))) / 2.0), 0.636620)) > (1.0)) * 1.)) * 2.0))
    v["i476"] = 0.019800 * np.tanh(np.maximum(((np.where(data["POS_NAME_CONTRACT_STATUS_Active_MEAN"] > 0, (((data["POS_NAME_CONTRACT_STATUS_Signed_MEAN"]) + (data["POS_NAME_CONTRACT_STATUS_Demand_MEAN"])) / 2.0), ((
        (-1.0 * ((0.636620)))) / 2.0)))), ((np.where(data["POS_NAME_CONTRACT_STATUS_Signed_MEAN"] < 0, data["POS_NAME_CONTRACT_STATUS_Demand_MEAN"], data["POS_NAME_CONTRACT_STATUS_Active_MEAN"])))))
    v["i477"] = 0.029940 * np.tanh(np.where(data["POS_SK_DPD_MEAN"] < 0, np.maximum((((((data["POS_SK_DPD_MEAN"]) > (((data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"]) / 2.0))) * 1.))),
                                                                                    (((((((data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"]) / 2.0)) > (3.141593)) * 1.)))), (((3.0) < (data["POS_SK_DPD_MAX"])) * 1.)))
    v["i478"] = 0.099999 * np.tanh((((((((data["POS_SK_DPD_DEF_MEAN"]) > (np.maximum(((data["POS_NAME_CONTRACT_STATUS_Signed_MEAN"])), (((-1.0 * ((data["POS_SK_DPD_MAX"]))))))))
                                        * 1.)) * 2.0)) * (np.where(data["POS_SK_DPD_DEF_MEAN"] > 0, (4.33465480804443359), data["POS_NAME_CONTRACT_STATUS_Signed_MEAN"]))))
    v["i479"] = 0.042010 * np.tanh((((((data["POS_NAME_CONTRACT_STATUS_Signed_MEAN"]) > (np.where(data["POS_NAME_CONTRACT_STATUS_Active_MEAN"] > 0, (((np.minimum(((data["POS_SK_DPD_MEAN"])), ((
        data["POS_NAME_CONTRACT_STATUS_Active_MEAN"])))) > (data["POS_SK_DPD_MAX"])) * 1.), (10.13535594940185547)))) * 1.)) + (((np.tanh((data["POS_SK_DPD_MEAN"]))) / 2.0))))
    v["i480"] = 0.098000 * np.tanh(np.where(data["INSTAL_PAYMENT_PERC_MEAN"] > 0, np.where((-1.0 * ((data["INSTAL_PAYMENT_PERC_MAX"]))) < 0, data[
                                   "INSTAL_DBD_MAX"], -2.0), np.where(data["INSTAL_PAYMENT_PERC_MAX"] < 0, (((data["INSTAL_DBD_MAX"]) > ((1.69303214550018311))) * 1.), 2.0)))
    v["i481"] = 0.099996 * np.tanh((-1.0 * ((np.where((((data["REFUSED_CNT_PAYMENT_MEAN"]) < (((data["AMT_ANNUITY"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) * 1.)
                                                      > 0, 0.0, ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (np.maximum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), ((((data["REFUSED_CNT_PAYMENT_MEAN"]) * 2.0)))))))))))
    v["i482"] = 0.099910 * np.tanh((((((((3.141593) < (data["AMT_CREDIT"])) * 1.)) * (np.where(data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]
                                                                                               < -99998, ((data["NEW_ANNUITY_TO_INCOME_RATIO"]) - (3.141593)), (4.74292945861816406))))) * ((4.74292945861816406))))
    v["i483"] = 0.056399 * np.tanh((-1.0 * (((((-1.0) > (np.where(data["CLOSED_MONTHS_BALANCE_MIN_MIN"] < -99998, ((((data["AMT_ANNUITY"]) * (
        np.maximum(((-1.0)), ((data["AMT_ANNUITY"])))))) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"])), data["NEW_CREDIT_TO_INCOME_RATIO"]))) * 1.)))))
    v["i484"] = 0.097201 * np.tanh((((((((np.where(data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"] > 0, 0.0, 2.0)) < (((np.where((-1.0 * (
        (data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]))) > 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], data["AMT_INCOME_TOTAL"])) * (data["CODE_GENDER"])))) * 1.)) * 2.0)) * 2.0))
    v["i485"] = 0.013002 * np.tanh(np.where(data["CLOSED_MONTHS_BALANCE_MIN_MIN"] < -99998, 0.0, np.minimum((((((((data["CLOSED_MONTHS_BALANCE_MIN_MIN"]) < (((data["BURO_MONTHS_BALANCE_MAX_MAX"]) * 2.0)))
                                                                                                                 * 1.)) - ((-1.0 * ((data["BURO_STATUS_2_MEAN_MEAN"]))))))), (((((data["BURO_STATUS_4_MEAN_MEAN"]) < (data["CLOSED_MONTHS_BALANCE_MIN_MIN"])) * 1.))))))
    v["i486"] = 0.099849 * np.tanh(((((((((data["BURO_MONTHS_BALANCE_SIZE_MEAN"]) < (((((((-1.0) + (data["CLOSED_MONTHS_BALANCE_MAX_MAX"])) / 2.0)) > (data["CLOSED_MONTHS_BALANCE_MIN_MIN"])) * 1.))) * 1.))
                                       < (data["CLOSED_MONTHS_BALANCE_MAX_MAX"])) * 1.)) * ((-1.0 * (((((data["CLOSED_MONTHS_BALANCE_MAX_MAX"]) < (data["BURO_MONTHS_BALANCE_SIZE_MEAN"])) * 1.)))))))
    v["i487"] = 0.099602 * np.tanh(((np.minimum((((((((3.0) / 2.0)) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.))), ((np.minimum(
        (((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) > (0.318310)) * 1.))), ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)))))))) / 2.0))
    v["i488"] = 0.098904 * np.tanh((((((((data["INSTAL_PAYMENT_PERC_MAX"]) * 2.0)) > (((data["INSTAL_PAYMENT_PERC_MEAN"]) + (data["NAME_INCOME_TYPE_Student"])))) * 1.)) - (
        (((data["INSTAL_PAYMENT_PERC_MAX"]) > (((((data["NAME_INCOME_TYPE_Student"]) * (0.636620))) + (data["NAME_INCOME_TYPE_Student"])))) * 1.))))
    v["i489"] = 0.100000 * np.tanh((((((-1.0) > (((np.maximum(((((data["INSTAL_DAYS_ENTRY_PAYMENT_SUM"]) * ((5.0))))), ((np.maximum(((0.636620)), ((
        ((data["NAME_INCOME_TYPE_Student"]) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))))))) - (data["INSTAL_AMT_INSTALMENT_MAX"])))) * 1.)) * 2.0))
    v["i490"] = 0.081300 * np.tanh(((2.0) * ((-1.0 * (((((np.where(data["AMT_ANNUITY"] < 0, data["CC_CNT_INSTALMENT_MATURE_CUM_VAR"], data["CC_AMT_DRAWINGS_CURRENT_VAR"])) > (
        np.where(data["AMT_ANNUITY"] < 0, (((data["AMT_ANNUITY"]) < (data["CC_AMT_DRAWINGS_CURRENT_VAR"])) * 1.), data["CC_AMT_PAYMENT_TOTAL_CURRENT_VAR"]))) * 1.)))))))
    v["i491"] = 0.084964 * np.tanh((((3.141593) < (np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"] > 0, np.where(((data["CC_AMT_DRAWINGS_POS_CURRENT_MAX"]) + (
        ((data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"]) * 2.0))) > 0, data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"], (14.65441703796386719)), ((data["CC_AMT_DRAWINGS_POS_CURRENT_MAX"]) + (1.0))))) * 1.))
    v["i492"] = 0.084899 * np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"] < 0, ((np.minimum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), ((((np.where(data["NAME_HOUSING_TYPE_Rented_apartment"] > 0, data[
                                   "NEW_CREDIT_TO_ANNUITY_RATIO"], data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])) * (data["REG_CITY_NOT_LIVE_CITY"])))))) * (np.tanh((data["NAME_HOUSING_TYPE_Rented_apartment"])))), data["NAME_HOUSING_TYPE_Rented_apartment"]))
    v["i493"] = 0.099903 * np.tanh(((((((((-1.0 * (((((data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]) > (data["NAME_HOUSING_TYPE_Rented_apartment"])) * 1.))))) < (data["LIVINGAREA_MODE"])) * 1.)) * (
        np.where((((data["LIVINGAREA_MEDI"]) + (data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"])) / 2.0) < 0, data["LIVINGAREA_MODE"], data["NAME_HOUSING_TYPE_Rented_apartment"])))) * 2.0))
    v["i494"] = 0.079029 * np.tanh((((((0.318310) < (np.where(data["LIVINGAREA_MEDI"] < -99998, data["FLOORSMAX_MODE"], (((((data["LIVINGAREA_MEDI"]) < (
        np.where(data["LIVINGAREA_MEDI"] > 0, data["FLOORSMAX_MODE"], data["LIVINGAREA_AVG"]))) * 1.)) - (data["LIVINGAREA_AVG"]))))) * 1.)) / 2.0))
    v["i495"] = 0.093600 * np.tanh(np.where(data["NEW_INC_BY_ORG"] < 0, ((np.where(data["AMT_ANNUITY"] > 0, data["REG_CITY_NOT_LIVE_CITY"], np.where(data["OCCUPATION_TYPE_Laborers"] > 0, data[
                                   "REG_CITY_NOT_LIVE_CITY"], data["NAME_INCOME_TYPE_Student"]))) * 2.0), (-1.0 * (((((data["REG_CITY_NOT_LIVE_CITY"]) > (data["NEW_INC_BY_ORG"])) * 1.))))))
    v["i496"] = 0.097989 * np.tanh(np.where((((data["INSTAL_DAYS_ENTRY_PAYMENT_SUM"]) > (((0.318310) * 2.0))) * 1.) > 0, (((((data["INSTAL_PAYMENT_PERC_MAX"]) > (data["INSTAL_PAYMENT_PERC_MEAN"])) * 1.))
                                                                                                                          * 2.0), np.maximum((((((data["INSTAL_PAYMENT_PERC_MEAN"]) + (data["INSTAL_DAYS_ENTRY_PAYMENT_SUM"])) / 2.0))), ((data["NAME_HOUSING_TYPE_Rented_apartment"])))))
    v["i497"] = 0.094620 * np.tanh((-1.0 * (((((((((2.0) < (np.where(data["CC_AMT_DRAWINGS_POS_CURRENT_MAX"] < 0, data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"], np.where(data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"]
                                                                                                                                                                     < 0, np.maximum(((data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"])), ((data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"]))), data["CC_AMT_DRAWINGS_POS_CURRENT_MAX"])))) * 1.)) * 2.0)) * 2.0)))))
    v["i498"] = 0.099922 * np.tanh((((data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"]) > (((data["CC_AMT_DRAWINGS_POS_CURRENT_MEAN"]) + (np.where(data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"] < 0, ((
        (data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"]) < (data["CC_CNT_DRAWINGS_OTHER_CURRENT_MAX"])) * 1.), (((data["CC_AMT_DRAWINGS_POS_CURRENT_MEAN"]) > (data["CC_AMT_DRAWINGS_OTHER_CURRENT_MAX"])) * 1.)))))) * 1.))
    v["i499"] = 0.090654 * np.tanh(((((np.minimum(((np.where(data["AMT_ANNUITY"] < 0, data["NAME_EDUCATION_TYPE_Higher_education"], np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] < 0, data[
                                   "OCCUPATION_TYPE_Laborers"], data["AMT_ANNUITY"])))), ((np.where(data["AMT_CREDIT"] < 0, data["OCCUPATION_TYPE_Laborers"], data["NAME_EDUCATION_TYPE_Higher_education"]))))) / 2.0)) / 2.0))
    v["i500"] = 0.099320 * np.tanh(((((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * 2.0)) * 2.0)) * 2.0)) * ((-1.0 * (((((np.where(np.minimum(
        ((data["CC_CNT_DRAWINGS_OTHER_CURRENT_MEAN"])), ((data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"]))) > 0, 3.141593, data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"])) > (2.0)) * 1.)))))))
    v["i501"] = 0.100000 * np.tanh((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (np.where(data["AMT_ANNUITY"] > 0, ((0.636620) + (data["AMT_ANNUITY"])), data[
                                   "NEW_CREDIT_TO_ANNUITY_RATIO"])))) > (3.0)) * 1.)) - (((data["ORGANIZATION_TYPE_Industry__type_2"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))
    v["i502"] = 0.098000 * np.tanh((-1.0 * (((((((((data["ORGANIZATION_TYPE_Industry__type_2"]) + (data["ORGANIZATION_TYPE_Industry__type_2"]))) + (
        (((data["NAME_INCOME_TYPE_Student"]) + (data["ORGANIZATION_TYPE_Industry__type_2"])) / 2.0)))) > ((-1.0 * ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (2.0))))))) * 1.)))))
    v["i503"] = 0.096000 * np.tanh((((((2.0) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)) * (np.where(((((2.13039326667785645)) < (
        data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.) > 0, (((7.0)) * (np.minimum(((data["NONLIVINGAREA_MODE"])), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))), (6.0)))))
    v["i504"] = 0.098781 * np.tanh(((((((((((4.0)) + ((((data["AMT_ANNUITY"]) > (((((-1.0 * ((data["NONLIVINGAREA_MEDI"])))) < (
        data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.))) * 1.))) / 2.0)) + (1.570796)) / 2.0)) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.))
    v["i505"] = 0.010101 * np.tanh((((((np.where(data["NONLIVINGAREA_AVG"] > 0, ((np.minimum(((((0.636620) + (0.318310)))), ((data["NEW_ANNUITY_TO_INCOME_RATIO"])))) - (
        data["NONLIVINGAREA_MODE"])), data["NONLIVINGAREA_AVG"])) - (data["NONLIVINGAREA_MODE"]))) > (0.318310)) * 1.))
    v["i506"] = 0.099900 * np.tanh(((3.141593) * (((3.141593) * (((((((2.79250931739807129)) < (data["LIVINGAPARTMENTS_MEDI"])) * 1.))
                                                                  * (((data["AMT_ANNUITY"]) * (((data["NONLIVINGAREA_MODE"]) - ((10.83971786499023438))))))))))))
    v["i507"] = 0.097990 * np.tanh((((((((data["LIVINGAPARTMENTS_MEDI"]) > (((3.0) + (np.where((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (np.maximum(
        ((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((data["NONLIVINGAREA_AVG"]))))) / 2.0) < 0, data["NONLIVINGAREA_MODE"], data["LIVINGAPARTMENTS_MEDI"]))))) * 1.)) * 2.0)) * 2.0))
    v["i508"] = 0.098996 * np.tanh((-1.0 * (((((0.636620) < (np.where(data["LIVINGAPARTMENTS_MEDI"] > 0, data["LIVINGAPARTMENTS_MODE"], (((((np.maximum(
        ((data["LIVINGAPARTMENTS_MEDI"])), ((data["NONLIVINGAPARTMENTS_MODE"])))) + ((((data["LIVINGAPARTMENTS_MEDI"]) > (data["NONLIVINGAPARTMENTS_MODE"])) * 1.))) / 2.0)) * 2.0)))) * 1.)))))
    v["i509"] = 0.056190 * np.tanh((((((((data["HOUSETYPE_MODE_block_of_flats"]) > (np.where(np.where(data["LIVINGAREA_AVG"] < -99998, 3.0, data[
                                   "HOUSETYPE_MODE_block_of_flats"]) < 0, ((data["LIVINGAREA_MODE"]) * 2.0), 3.0))) * 1.)) - (data["NAME_INCOME_TYPE_Student"]))) * 2.0))
    v["i510"] = 0.095000 * np.tanh((((((((data["FLOORSMIN_AVG"]) > (((data["HOUSETYPE_MODE_block_of_flats"]) + ((((data["FLOORSMIN_MEDI"]) + (
        np.where(data["LIVINGAREA_MODE"] < -99998, data["HOUSETYPE_MODE_block_of_flats"], np.tanh((np.tanh((-1.0))))))) / 2.0))))) * 1.)) * 2.0)) * 2.0))
    v["i511"] = 0.056400 * np.tanh(((data["HOUSETYPE_MODE_specific_housing"]) * (((((np.where(data["WALLSMATERIAL_MODE_Panel"] < 0, (-1.0 * ((np.where(data["HOUSETYPE_MODE_specific_housing"] < 0, data[
                                   "FONDKAPREMONT_MODE_reg_oper_account"], data["HOUSETYPE_MODE_specific_housing"])))), data["FONDKAPREMONT_MODE_reg_oper_account"])) + (data["ORGANIZATION_TYPE_Trade__type_4"]))) + (data["NAME_INCOME_TYPE_Student"])))))
    return Output(v.sum(axis=1))


def GP2(data):
    v = pd.DataFrame()
    v["i0"] = 0.059995 * np.tanh(((((((((((data["DAYS_BIRTH"]) + (data["REGION_RATING_CLIENT_W_CITY"]))) * 2.0)) + (((np.where(
        data["REFUSED_AMT_CREDIT_MAX"] < -99998, -1.0, 3.141593)) + (data["NAME_INCOME_TYPE_Working"]))))) + (data["DAYS_BIRTH"]))) * 2.0))
    v["i1"] = 0.095000 * np.tanh(((((((((np.where(data["NAME_EDUCATION_TYPE_Higher_education"] < 0, np.maximum(((data["REGION_RATING_CLIENT_W_CITY"])), ((
        ((1.0) + (data["REFUSED_DAYS_DECISION_MEAN"]))))), data["REFUSED_DAYS_DECISION_MEAN"])) - (data["CODE_GENDER"]))) + (data["DAYS_BIRTH"]))) * 2.0)) * 2.0))
    v["i2"] = 0.098800 * np.tanh((((((((((-1.0 * ((np.tanh(((((((1.0)) / 2.0)) - (data["CLOSED_DAYS_CREDIT_MEAN"])))))))) - (
        data["CLOSED_AMT_CREDIT_SUM_MEAN"]))) * 2.0)) - (((0.636620) - (data["CLOSED_DAYS_CREDIT_MEAN"]))))) * 2.0))
    v["i3"] = 0.097103 * np.tanh(((data["REGION_RATING_CLIENT_W_CITY"]) + (((((np.tanh((data["REFUSED_DAYS_DECISION_MAX"]))) + (((data["REGION_RATING_CLIENT"]) + (
        ((np.maximum(((((data["DAYS_BIRTH"]) - (data["NAME_EDUCATION_TYPE_Higher_education"])))), ((data["REFUSED_CNT_PAYMENT_SUM"])))) * 2.0)))))) * 2.0))))
    v["i4"] = 0.099996 * np.tanh(((np.tanh((data["REFUSED_DAYS_DECISION_MEAN"]))) + (((np.maximum(((((data["DAYS_ID_PUBLISH"]) + (data["NAME_INCOME_TYPE_Working"])))),
                                                                                                  ((data["REFUSED_CNT_PAYMENT_SUM"])))) + (((((data["DAYS_BIRTH"]) - (data["CODE_GENDER"]))) + (data["REGION_RATING_CLIENT_W_CITY"])))))))
    v["i5"] = 0.099985 * np.tanh(((((np.maximum(((data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"])), ((((((np.where(data["NEW_SCORES_STD"] < 0, data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"], data["NEW_DOC_IND_KURT"])) + (
        data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]))) + (np.maximum(((data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"])), ((data["NEW_SCORES_STD"]))))))))) * 2.0)) * 2.0))
    v["i6"] = 0.099930 * np.tanh(((((((((((data["REG_CITY_NOT_LIVE_CITY"]) / 2.0)) + (data["REGION_RATING_CLIENT_W_CITY"]))) + (
        ((((data["FLAG_DOCUMENT_3"]) + (data["DAYS_ID_PUBLISH"]))) * 2.0)))) + (((data["REG_CITY_NOT_WORK_CITY"]) - (data["EMERGENCYSTATE_MODE_No"]))))) * 2.0))
    v["i7"] = 0.099976 * np.tanh((((7.0)) * (np.where(data["CODE_GENDER"] < 0, 3.0, ((((((((data["REFUSED_AMT_APPLICATION_MAX"]) < (
        data["REFUSED_CNT_PAYMENT_SUM"])) * 1.)) + (((data["REFUSED_CNT_PAYMENT_SUM"]) * 2.0)))) + (((data["REFUSED_DAYS_DECISION_MAX"]) / 2.0))) / 2.0)))))
    v["i8"] = 0.099595 * np.tanh(((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) - (np.where(data["NAME_EDUCATION_TYPE_Higher_education"] > 0, data["FLAG_OWN_CAR"], np.where(np.where(
        data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"] > 0, data["NAME_EDUCATION_TYPE_Higher_education"], data["NEW_CAR_TO_BIRTH_RATIO"]) > 0, (3.0), ((data["CLOSED_DAYS_CREDIT_VAR"]) - ((11.01282405853271484))))))))
    v["i9"] = 0.099984 * np.tanh(((((data["REG_CITY_NOT_LIVE_CITY"]) + (np.where(data["NEW_DOC_IND_KURT"] > 0, ((np.minimum(((data["REG_CITY_NOT_LIVE_CITY"])), ((
        data["ACTIVE_DAYS_CREDIT_VAR"])))) - (data["NEW_CAR_TO_BIRTH_RATIO"])), -2.0)))) + ((((data["REGION_RATING_CLIENT_W_CITY"]) + (data["NEW_DOC_IND_KURT"])) / 2.0))))
    v["i10"] = 0.100000 * np.tanh(((((((data["NEW_SCORES_STD"]) - (np.where((((data["CLOSED_AMT_CREDIT_SUM_SUM"]) + ((((data["CLOSED_AMT_CREDIT_SUM_SUM"]) + (
        np.tanh((1.0)))) / 2.0))) / 2.0) > 0, (1.0), data["NAME_FAMILY_STATUS_Married"])))) * 2.0)) - (data["CLOSED_AMT_CREDIT_SUM_MEAN"])))
    v["i11"] = 0.099915 * np.tanh((((((((((data["ACTIVE_DAYS_CREDIT_VAR"]) > (data["EXT_SOURCE_1"])) * 1.)) - (((np.where(data["EXT_SOURCE_1"] < -99998, np.tanh(
        (data["ACTIVE_DAYS_CREDIT_VAR"])), data["EXT_SOURCE_1"])) - (np.tanh((data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"]))))))) * 2.0)) * 2.0))
    v["i12"] = 0.099796 * np.tanh(((((data["DAYS_BIRTH"]) - (data["NAME_INCOME_TYPE_State_servant"]))) + ((((((data["DAYS_REGISTRATION"]) + (np.where(
        data["NAME_FAMILY_STATUS_Married"] > 0, (((data["ACTIVE_DAYS_CREDIT_VAR"]) + (2.0)) / 2.0), 1.570796))) / 2.0)) - (data["WALLSMATERIAL_MODE_Panel"])))))
    v["i13"] = 0.100000 * np.tanh((((((((-1.0 * ((((((np.maximum(((data["EXT_SOURCE_1"])), ((data["APPROVED_AMT_DOWN_PAYMENT_MAX"])))) * 2.0)) * 2.0))))
                                        ) - (np.maximum(((((data["OCCUPATION_TYPE_Core_staff"]) / 2.0))), ((0.318310)))))) * 2.0)) * 2.0))
    v["i14"] = 0.099680 * np.tanh(((data["NEW_DOC_IND_KURT"]) + (((((((data["NEW_DOC_IND_KURT"]) + (np.tanh((np.where(data["REFUSED_AMT_CREDIT_MEAN"] > 0, data[
                                  "NEW_DOC_IND_KURT"], ((data["REFUSED_DAYS_DECISION_MEAN"]) * 2.0))))))) - (data["REFUSED_AMT_CREDIT_MEAN"]))) + (data["REFUSED_CNT_PAYMENT_SUM"])))))
    v["i15"] = 0.099997 * np.tanh(((((((data["OCCUPATION_TYPE_Low_skill_Laborers"]) + (data["REG_CITY_NOT_LIVE_CITY"]))) - (data["FLAG_PHONE"]))) + (np.where(
        data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"] < -99998, data["DAYS_REGISTRATION"], ((data["ORGANIZATION_TYPE_Self_employed"]) + (((data["NEW_SCORES_STD"]) + (data["REG_CITY_NOT_LIVE_CITY"]))))))))
    v["i16"] = 0.099925 * np.tanh(np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"] > 0, 3.141593, ((data["REGION_RATING_CLIENT"]) - (np.where(
        (((np.tanh((data["APPROVED_AMT_DOWN_PAYMENT_MEAN"]))) < (data["APPROVED_AMT_DOWN_PAYMENT_MAX"])) * 1.) > 0, (4.07468795776367188), data["APPROVED_AMT_DOWN_PAYMENT_MAX"])))))
    v["i17"] = 0.099600 * np.tanh(((np.minimum(((data["NEW_CREDIT_TO_INCOME_RATIO"])), ((data["ORGANIZATION_TYPE_Transport__type_3"])))) + ((((np.where(data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"] < 0, ((
        data["FLAG_WORK_PHONE"]) + (((data["ORGANIZATION_TYPE_Transport__type_3"]) + (data["ORGANIZATION_TYPE_Business_Entity_Type_3"])))), (13.65371227264404297))) + (data["ORGANIZATION_TYPE_Construction"])) / 2.0))))
    v["i18"] = 0.099990 * np.tanh((-1.0 * ((np.where(((np.maximum(((np.maximum(((data["OCCUPATION_TYPE_Accountants"])), ((data["EXT_SOURCE_1"]))))), ((np.maximum(
        ((data["NEW_CAR_TO_EMPLOY_RATIO"])), ((data["NEW_CAR_TO_EMPLOY_RATIO"]))))))) + (data["OCCUPATION_TYPE_High_skill_tech_staff"])) < 0, data["EXT_SOURCE_1"], (13.81985569000244141))))))
    v["i19"] = 0.095400 * np.tanh(((((data["OCCUPATION_TYPE_Drivers"]) - (data["OCCUPATION_TYPE_Core_staff"]))) + (np.maximum(
        ((data["ACTIVE_AMT_ANNUITY_MAX"])), ((np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, -1.0, (-1.0 * ((data["ACTIVE_DAYS_CREDIT_VAR"]))))))))))
    v["i20"] = 0.099600 * np.tanh(np.where((((data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]) > (data["NAME_INCOME_TYPE_Unemployed"])) * 1.) > 0, (9.65223503112792969), np.where(
        ((data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]) - (data["DAYS_ID_PUBLISH"])) < -99998, np.maximum(((data["DAYS_ID_PUBLISH"])), ((data["NEW_DOC_IND_KURT"]))), -2.0)))
    v["i21"] = 0.099870 * np.tanh(((data["INSTAL_AMT_INSTALMENT_MAX"]) + (((((((((np.where((((data["INSTAL_AMT_INSTALMENT_MAX"]) + (
        ((1.0) / 2.0))) / 2.0) > 0, data["INSTAL_AMT_INSTALMENT_MAX"], 1.570796)) * 2.0)) * 2.0)) * 2.0)) - (1.570796)))))
    v["i22"] = 0.099732 * np.tanh(((np.maximum(((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"])), ((np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"] < -99998, np.where(((((0.318310) + (
        data["NEW_CREDIT_TO_INCOME_RATIO"]))) + (data["NEW_DOC_IND_KURT"])) > 0, 0.318310, data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"]), data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"]))))) * 2.0))
    v["i23"] = 0.099959 * np.tanh((((-1.0 * ((((np.where(data["CC_CNT_INSTALMENT_MATURE_CUM_VAR"] < -99998, data["ORGANIZATION_TYPE_Military"], -1.0)) * 2.0))))) - (
        ((((((((((-1.0) + (data["CC_CNT_INSTALMENT_MATURE_CUM_VAR"])) / 2.0)) < (data["CC_MONTHS_BALANCE_VAR"])) * 1.)) * 2.0)) * 2.0))))
    v["i24"] = 0.099950 * np.tanh(np.maximum(((((data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"]) + (data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"])))), ((np.where(
        (((data["CC_AMT_DRAWINGS_ATM_CURRENT_MAX"]) < (((1.570796) * (data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"])))) * 1.) > 0, 3.141593, (-1.0 * ((0.318310))))))))
    v["i25"] = 0.099964 * np.tanh(np.where(np.maximum(((data["NAME_EDUCATION_TYPE_Lower_secondary"])), (((((((0.318310) / 2.0)) > ((-1.0 * ((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]))))) * 1.))))
                                           > 0, (11.36913585662841797), (-1.0 * (((((((data["NAME_FAMILY_STATUS_Married"]) > (data["NAME_EDUCATION_TYPE_Lower_secondary"])) * 1.)) / 2.0))))))
    v["i26"] = 0.097999 * np.tanh(np.minimum(((((((((1.570796) > (((0.318310) + (data["HOUR_APPR_PROCESS_START"])))) * 1.)) + (
        data["OCCUPATION_TYPE_Laborers"])) / 2.0))), (((-1.0 * ((((data["CC_NAME_CONTRACT_STATUS_Approved_VAR"]) + (data["HOUR_APPR_PROCESS_START"])))))))))
    v["i27"] = 0.099000 * np.tanh(((((((((np.where((((data["CC_AMT_DRAWINGS_ATM_CURRENT_MAX"]) < (((np.tanh((np.tanh((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]))))) + (
        data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"])))) * 1.) > 0, 2.0, data["NAME_INCOME_TYPE_Unemployed"])) * 2.0)) * 2.0)) * 2.0)) * 2.0))
    v["i28"] = 0.097996 * np.tanh((((((np.maximum(((data["OCCUPATION_TYPE_Low_skill_Laborers"])), (((((((np.maximum(((data["ORGANIZATION_TYPE_Realtor"])), ((data["NAME_INCOME_TYPE_Unemployed"])))) + (
        data["ORGANIZATION_TYPE_Transport__type_3"])) / 2.0)) + (data["NAME_INCOME_TYPE_Maternity_leave"])))))) + (data["ORGANIZATION_TYPE_Trade__type_3"])) / 2.0)) - (data["ORGANIZATION_TYPE_Industry__type_9"])))
    v["i29"] = 0.098000 * np.tanh(((((np.tanh((np.tanh(((((data["ORGANIZATION_TYPE_Self_employed"]) + (np.tanh((data["APPROVED_RATE_DOWN_PAYMENT_MAX"])))) / 2.0)))))) - (
        np.maximum(((((data["ORGANIZATION_TYPE_Military"]) * 2.0))), ((data["APPROVED_RATE_DOWN_PAYMENT_MAX"])))))) - (np.tanh((data["APPROVED_AMT_DOWN_PAYMENT_MAX"])))))
    v["i30"] = 0.097500 * np.tanh(((((((data["ORGANIZATION_TYPE_Realtor"]) + (((data["NAME_INCOME_TYPE_Unemployed"]) + (data["ORGANIZATION_TYPE_Restaurant"]))))) - (np.maximum(
        ((data["ORGANIZATION_TYPE_Trade__type_2"])), ((data["OCCUPATION_TYPE_Medicine_staff"])))))) - (np.maximum(((data["ORGANIZATION_TYPE_Trade__type_2"])), ((data["ORGANIZATION_TYPE_Security_Ministries"]))))))
    v["i31"] = 0.098960 * np.tanh(((((((data["NAME_INCOME_TYPE_Unemployed"]) + ((((data["REFUSED_DAYS_DECISION_MEAN"]) < (np.where(data["REFUSED_DAYS_DECISION_MAX"] < 0, data[
                                  "REFUSED_DAYS_DECISION_MAX"], (((np.tanh((data["REFUSED_DAYS_DECISION_MEAN"]))) < (data["REFUSED_DAYS_DECISION_MAX"])) * 1.)))) * 1.)))) * ((10.95812892913818359)))) * 2.0))
    v["i32"] = 0.099500 * np.tanh(((((((((data["NAME_HOUSING_TYPE_Municipal_apartment"]) + (-2.0)) / 2.0)) * (np.maximum(((np.maximum(((data["CC_CNT_INSTALMENT_MATURE_CUM_VAR"])), ((data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"]))))),
                                                                                                                         ((data["NAME_HOUSING_TYPE_Municipal_apartment"])))))) + ((-1.0 * (((((data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"]) > (data["CC_CNT_INSTALMENT_MATURE_CUM_VAR"])) * 1.)))))) / 2.0))
    v["i33"] = 0.096799 * np.tanh(np.maximum((((((0.636620) + (np.maximum(((-1.0)), ((np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"] < 0, data["CC_CNT_DRAWINGS_OTHER_CURRENT_MAX"], data[
                                  "CC_CNT_DRAWINGS_POS_CURRENT_MAX"])))))) / 2.0))), ((((((data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"]) + (-1.0))) + (data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]))))))
    v["i34"] = 0.099997 * np.tanh(np.minimum(((((np.maximum(((data["NEW_SCORES_STD"])), ((data["DAYS_REGISTRATION"])))) / 2.0))), ((
        ((((((0.636620) - ((-1.0 * ((((((data["NEW_CREDIT_TO_INCOME_RATIO"]) + (0.636620))) * 2.0))))))) * 2.0)) * 2.0)))))
    v["i35"] = 0.099015 * np.tanh(((np.minimum(((np.maximum(((((((data["NAME_EDUCATION_TYPE_Incomplete_higher"]) * (data["AMT_INCOME_TOTAL"]))) * 2.0))), ((data["NAME_EDUCATION_TYPE_Incomplete_higher"]))))), ((
        data["AMT_INCOME_TOTAL"])))) - (((((((data["NAME_EDUCATION_TYPE_Incomplete_higher"]) + (data["WEEKDAY_APPR_PROCESS_START_SUNDAY"])) / 2.0)) + (data["ORGANIZATION_TYPE_Bank"])) / 2.0))))
    v["i36"] = 0.098830 * np.tanh(np.minimum(((1.570796)), (((((((((-1.0 * ((data["AMT_CREDIT"])))) * (
        ((((((data["AMT_CREDIT"]) * 2.0)) * 2.0)) - (data["ORGANIZATION_TYPE_Military"]))))) - (data["AMT_CREDIT"]))) + (1.570796))))))
    v["i37"] = 0.098020 * np.tanh((((((data["NEW_CREDIT_TO_INCOME_RATIO"]) > (((((3.0) - (data["NAME_INCOME_TYPE_Unemployed"]))) / 2.0))) * 1.)) - (
        ((((((data["CC_CNT_INSTALMENT_MATURE_CUM_VAR"]) + ((-1.0 * ((-1.0))))) / 2.0)) > (data["NEW_CREDIT_TO_INCOME_RATIO"])) * 1.))))
    v["i38"] = 0.099905 * np.tanh(((np.maximum(((np.maximum((((((data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"]) < (data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"])) * 1.))), ((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]))))),
                                               (((((data["AMT_INCOME_TOTAL"]) > ((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) < (np.tanh((((data["CC_CNT_DRAWINGS_POS_CURRENT_MEAN"]) * 2.0))))) * 1.))) * 1.))))) * 2.0))
    v["i39"] = 0.097452 * np.tanh((((((data["NAME_INCOME_TYPE_Maternity_leave"]) + (((((-1.0 * ((data["WEEKDAY_APPR_PROCESS_START_SATURDAY"])))) + (((((((data["ORGANIZATION_TYPE_Realtor"]) + (
        data["NAME_INCOME_TYPE_Unemployed"]))) + (data["ORGANIZATION_TYPE_Transport__type_3"]))) - (data["ORGANIZATION_TYPE_Military"])))) / 2.0)))) + (data["ORGANIZATION_TYPE_Transport__type_3"])) / 2.0))
    v["i40"] = 0.098003 * np.tanh((((((((-1.0 * ((data["WEEKDAY_APPR_PROCESS_START_MONDAY"])))) + (((np.where(data["CC_CNT_INSTALMENT_MATURE_CUM_VAR"] < -99998, data["ORGANIZATION_TYPE_Construction"], (10.0)))
                                                                                                    * (data["ORGANIZATION_TYPE_Construction"])))) / 2.0)) + ((-1.0 * ((np.maximum(((data["ORGANIZATION_TYPE_Police"])), ((data["CC_CNT_INSTALMENT_MATURE_CUM_VAR"])))))))) / 2.0))
    v["i41"] = 0.100000 * np.tanh(np.where(data["PREV_AMT_ANNUITY_MEAN"] < -99998, np.tanh((data["PREV_APP_CREDIT_PERC_MEAN"])), ((data["AMT_INCOME_TOTAL"]) - (
        ((data["PREV_AMT_ANNUITY_MEAN"]) + (np.tanh((((data["PREV_APP_CREDIT_PERC_MEAN"]) * (((3.141593) + (data["PREV_APP_CREDIT_PERC_MAX"]))))))))))))
    v["i42"] = 0.099499 * np.tanh((((((((((data["OCCUPATION_TYPE_Sales_staff"]) + (data["NAME_INCOME_TYPE_Maternity_leave"])) / 2.0)) + (((data["ORGANIZATION_TYPE_Trade__type_7"]) - (
        data["ORGANIZATION_TYPE_Industry__type_12"])))) / 2.0)) + (((((data["NAME_INCOME_TYPE_Unemployed"]) + (data["OCCUPATION_TYPE_Cooking_staff"]))) - (data["ORGANIZATION_TYPE_Industry__type_9"])))) / 2.0))
    v["i43"] = 0.098011 * np.tanh(np.where(((data["ORGANIZATION_TYPE_School"]) * (data["CODE_GENDER"])) > 0, ((np.tanh((((data["CODE_GENDER"]) * (
        data["AMT_INCOME_TOTAL"]))))) - (data["ORGANIZATION_TYPE_School"])), (((data["OCCUPATION_TYPE_Security_staff"]) + (data["AMT_INCOME_TOTAL"])) / 2.0)))
    v["i44"] = 0.095501 * np.tanh((((((((data["ORGANIZATION_TYPE_Realtor"]) > ((-1.0 * ((((np.where(data["APPROVED_AMT_DOWN_PAYMENT_MAX"] < -99998, data["OCCUPATION_TYPE_Cleaning_staff"], data[
                                  "APPROVED_AMT_DOWN_PAYMENT_MEAN"])) - (data["APPROVED_AMT_DOWN_PAYMENT_MAX"]))))))) * 1.)) + (((data["OCCUPATION_TYPE_Cleaning_staff"]) * 2.0)))) + (data["OCCUPATION_TYPE_Cleaning_staff"])))
    v["i45"] = 0.099831 * np.tanh(((data["NAME_INCOME_TYPE_Unemployed"]) + (((((data["NAME_HOUSING_TYPE_Office_apartment"]) * (data["REG_REGION_NOT_LIVE_REGION"]))) + (
        ((data["HOUSETYPE_MODE_specific_housing"]) + (((data["NAME_CONTRACT_TYPE_Cash_loans"]) * (((data["AMT_INCOME_TOTAL"]) * 2.0))))))))))
    v["i46"] = 0.099521 * np.tanh((-1.0 * ((((np.where(data["NEW_CAR_TO_EMPLOY_RATIO"] < -99998, np.maximum(((data["FONDKAPREMONT_MODE_org_spec_account"])), ((data["ORGANIZATION_TYPE_Security_Ministries"]))),
                                                       np.where((((1.0) + (data["NEW_CREDIT_TO_INCOME_RATIO"])) / 2.0) < 0, data["FONDKAPREMONT_MODE_org_spec_account"], ((data["NEW_CAR_TO_EMPLOY_RATIO"]) * 2.0)))) * 2.0)))))
    v["i47"] = 0.099301 * np.tanh(((((3.0) * (np.where(data["ORGANIZATION_TYPE_Realtor"] > 0, 1.0, np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]
                                                                                                            < -99998, data["ORGANIZATION_TYPE_Legal_Services"], ((((((7.0)) * (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]))) + (2.0)) / 2.0)))))) * 2.0))
    v["i48"] = 0.098000 * np.tanh((-1.0 * ((np.where(data["AMT_REQ_CREDIT_BUREAU_QRT"] > 0, ((((3.141593) - (data["AMT_REQ_CREDIT_BUREAU_QRT"]))) - (
        data["AMT_REQ_CREDIT_BUREAU_QRT"])), np.tanh((np.tanh(((((data["AMT_REQ_CREDIT_BUREAU_YEAR"]) + ((-1.0 * ((data["NEW_CREDIT_TO_INCOME_RATIO"]))))) / 2.0))))))))))
    v["i49"] = 0.099600 * np.tanh(((((-1.0) + ((((data["POS_NAME_CONTRACT_STATUS_Signed_MEAN"]) + (data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"])) / 2.0)))) + (
        (((((((((data["POS_SK_DPD_MAX"]) > (((data["NAME_INCOME_TYPE_Unemployed"]) + (data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"])))) * 1.)) * 2.0)) * 2.0)) * 2.0))))
    v["i50"] = 0.095100 * np.tanh(((np.minimum((((((data["AMT_CREDIT"]) > (-1.0)) * 1.))), (((((data["AMT_CREDIT"]) < (
        np.where(data["AMT_CREDIT"] < 0, data["NAME_TYPE_SUITE_Other_B"], ((0.318310) / 2.0)))) * 1.))))) - (0.318310)))
    v["i51"] = 0.099453 * np.tanh(((((((np.where((((data["NAME_INCOME_TYPE_Unemployed"]) < (((data["BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"]) / 2.0)))
                                                  * 1.) > 0, 1.570796, ((((data["NAME_INCOME_TYPE_Unemployed"]) * 2.0)) * 2.0))) * 2.0)) * 2.0)) * 2.0))
    v["i52"] = 0.099700 * np.tanh(((np.where(data["AMT_INCOME_TOTAL"] < 0, (-1.0 * ((data["AMT_INCOME_TOTAL"]))), ((np.minimum((((-1.0 * ((np.minimum(
        ((data["AMT_INCOME_TOTAL"])), ((data["BURO_DAYS_CREDIT_VAR"])))))))), ((data["BURO_DAYS_CREDIT_VAR"])))) / 2.0))) * (data["NEW_CREDIT_TO_INCOME_RATIO"])))
    v["i53"] = 0.099200 * np.tanh((((((((data["NAME_HOUSING_TYPE_Rented_apartment"]) - (data["ORGANIZATION_TYPE_Industry__type_12"]))) - ((((data["NAME_FAMILY_STATUS_Married"]) + (np.where(
        data["NAME_FAMILY_STATUS_Married"] > 0, data["ORGANIZATION_TYPE_Transport__type_2"], data["NAME_INCOME_TYPE_Unemployed"]))) / 2.0)))) + ((-1.0 * ((((data["FLAG_OWN_REALTY"]) / 2.0)))))) / 2.0))
    v["i54"] = 0.090218 * np.tanh(((((data["FLAG_WORK_PHONE"]) * ((((((((data["ORGANIZATION_TYPE_Hotel"]) - (data["AMT_INCOME_TOTAL"]))) > (
        (((data["AMT_INCOME_TOTAL"]) + (data["NEW_CREDIT_TO_INCOME_RATIO"])) / 2.0))) * 1.)) - ((((data["NEW_CREDIT_TO_INCOME_RATIO"]) + (data["AMT_INCOME_TOTAL"])) / 2.0)))))) / 2.0))
    v["i55"] = 0.099294 * np.tanh(np.where(data["ORGANIZATION_TYPE_XNA"] < 0, (((((data["NAME_EDUCATION_TYPE_Academic_degree"]) > (
        ((((2.0) * (data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]))) + (0.318310)))) * 1.)) - (0.318310)), (((data["NAME_EDUCATION_TYPE_Academic_degree"]) < (data["NEW_CREDIT_TO_INCOME_RATIO"])) * 1.)))
    v["i56"] = 0.086898 * np.tanh(np.where(data["ORGANIZATION_TYPE_Transport__type_4"] > 0, data["ORGANIZATION_TYPE_Transport__type_4"], np.where(data["AMT_INCOME_TOTAL"] > 0, ((
        data["AMT_INCOME_TOTAL"]) * (((data["ORGANIZATION_TYPE_Business_Entity_Type_3"]) - (data["AMT_INCOME_TOTAL"])))), ((data["ORGANIZATION_TYPE_Transport__type_4"]) - (data["NAME_FAMILY_STATUS_Widow"])))))
    v["i57"] = 0.088497 * np.tanh((((((data["ORGANIZATION_TYPE_Trade__type_3"]) + ((((((((((data["NAME_INCOME_TYPE_Unemployed"]) + (data["NAME_TYPE_SUITE_Spouse__partner"])) / 2.0)) + (data["NAME_INCOME_TYPE_Maternity_leave"])
                                                                                        ) / 2.0)) + ((-1.0 * ((data["NAME_TYPE_SUITE_Spouse__partner"]))))) / 2.0))) / 2.0)) - (np.maximum(((data["NAME_EDUCATION_TYPE_Academic_degree"])), ((data["ORGANIZATION_TYPE_Hotel"]))))))
    v["i58"] = 0.095795 * np.tanh((((((data["NAME_INCOME_TYPE_Unemployed"]) + (((((data["OCCUPATION_TYPE_Waiters_barmen_staff"]) / 2.0)) - (data["REG_REGION_NOT_LIVE_REGION"])))) / 2.0)) + (
        ((data["NAME_TYPE_SUITE_Other_B"]) + ((((data["NAME_TYPE_SUITE_Other_B"]) > (((data["REG_REGION_NOT_LIVE_REGION"]) - (data["CLOSED_AMT_ANNUITY_MEAN"])))) * 1.))))))
    v["i59"] = 0.097340 * np.tanh(((np.maximum(((((((np.where(data["BURO_STATUS_2_MEAN_MEAN"] < -99998, data["NAME_INCOME_TYPE_Maternity_leave"], data["BURO_STATUS_4_MEAN_MEAN"]))
                                                    * 2.0)) * 2.0))), ((((((np.maximum(((data["BURO_MONTHS_BALANCE_MIN_MIN"])), ((data["BURO_STATUS_2_MEAN_MEAN"])))) * 2.0)) - (3.0)))))) * 2.0))
    v["i60"] = 0.098002 * np.tanh((((((((((((data["ORGANIZATION_TYPE_Transport__type_3"]) * 2.0)) < (data["AMT_INCOME_TOTAL"])) * 1.)) + (((data["ORGANIZATION_TYPE_Transport__type_3"]) - (
        data["ORGANIZATION_TYPE_Electricity"])))) / 2.0)) + (np.maximum(((data["ORGANIZATION_TYPE_Legal_Services"])), (((((data["ORGANIZATION_TYPE_Legal_Services"]) < (data["CLOSED_AMT_ANNUITY_MAX"])) * 1.)))))) / 2.0))
    v["i61"] = 0.099951 * np.tanh((((((data["ORGANIZATION_TYPE_Industry__type_3"]) - (data["FLAG_CONT_MOBILE"]))) + ((((((data["NAME_INCOME_TYPE_Maternity_leave"]) + (
        (((data["NAME_INCOME_TYPE_Unemployed"]) + (data["NAME_CONTRACT_TYPE_Revolving_loans"])) / 2.0))) / 2.0)) - (data["NAME_EDUCATION_TYPE_Academic_degree"])))) / 2.0))
    v["i62"] = 0.100000 * np.tanh(((((np.where(data["OCCUPATION_TYPE_Accountants"] < 0, data["ORGANIZATION_TYPE_Police"], data["OCCUPATION_TYPE_Accountants"])) * (((data["FONDKAPREMONT_MODE_reg_oper_spec_account"]) + (
        ((data["NAME_INCOME_TYPE_Maternity_leave"]) - (data["ORGANIZATION_TYPE_Police"]))))))) + (((data["OCCUPATION_TYPE_Low_skill_Laborers"]) - (data["WALLSMATERIAL_MODE_Monolithic"])))))
    v["i63"] = 0.095961 * np.tanh(((np.maximum(((data["NAME_INCOME_TYPE_Unemployed"])), ((((np.maximum((((((data["OCCUPATION_TYPE_Security_staff"]) > (((data["NEW_CREDIT_TO_INCOME_RATIO"]) * (
        data["ORGANIZATION_TYPE_Realtor"])))) * 1.))), ((np.where(data["ORGANIZATION_TYPE_Realtor"] < 0, data["NAME_INCOME_TYPE_Maternity_leave"], data["NEW_CREDIT_TO_INCOME_RATIO"]))))) * 2.0))))) * 2.0))
    v["i64"] = 0.099960 * np.tanh(((((((np.where(((((data["NEW_SOURCES_PROD"]) + (data["NEW_SOURCES_PROD"]))) + (1.570796)) >
                                                 0, -2.0, np.where(data["NEW_SOURCES_PROD"] < -99998, data["NAME_INCOME_TYPE_Unemployed"], (10.0)))) * 2.0)) * 2.0)) * 2.0))
    v["i65"] = 0.099910 * np.tanh(np.where(((data["NEW_SOURCES_PROD"]) - (np.tanh((((data["EXT_SOURCE_1"]) * 2.0))))) > 0, -2.0, (-1.0 * (
        (np.where(data["NEW_SOURCES_PROD"] < -99998, data["NAME_EDUCATION_TYPE_Academic_degree"], ((((data["NEW_SOURCES_PROD"]) * 2.0)) * 2.0)))))))
    v["i66"] = 0.099640 * np.tanh(np.where(data["NEW_SOURCES_PROD"] < -99998, data["NAME_INCOME_TYPE_Maternity_leave"], ((
        ((((((-1.0) - (((data["NEW_SOURCES_PROD"]) * 2.0)))) - (data["NEW_SCORES_STD"]))) * 2.0)) - (np.minimum(((data["NEW_SOURCES_PROD"])), ((data["NEW_SCORES_STD"])))))))
    v["i67"] = 0.099950 * np.tanh((-1.0 * (((((((data["NEW_SOURCES_PROD"]) < (((data["REFUSED_AMT_GOODS_PRICE_MAX"]) + (1.570796)))) * 1.)) + (
        np.where(data["NEW_SOURCES_PROD"] < -99998, np.tanh((-2.0)), ((data["NEW_SOURCES_PROD"]) * 2.0))))))))
    v["i68"] = 0.099490 * np.tanh((-1.0 * ((np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])), ((np.where(data["NEW_SCORES_STD"] > 0, data[
                                  "DAYS_BIRTH"], (-1.0 * (((((((((((((data["DAYS_BIRTH"]) / 2.0)) / 2.0)) / 2.0)) / 2.0)) > (data["NAME_INCOME_TYPE_Maternity_leave"])) * 1.))))))))))))
    v["i69"] = 0.099500 * np.tanh(np.where(((((((np.maximum(((1.570796)), ((data["CLOSED_AMT_CREDIT_SUM_MEAN"])))) < (data["CLOSED_DAYS_CREDIT_MEAN"])) * 1.)) + (
        data["CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN"])) / 2.0) > 0, 3.0, (((data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"]) > (((1.0) + (data["CLOSED_DAYS_CREDIT_MEAN"])))) * 1.)))
    v["i70"] = 0.096400 * np.tanh(np.where((((data["NEW_CREDIT_TO_INCOME_RATIO"]) + (data["DAYS_BIRTH"])) / 2.0) > 0, ((np.where(
        data["DAYS_BIRTH"] > 0, ((2.0) - (data["DAYS_BIRTH"])), np.tanh((data["DAYS_BIRTH"])))) - (data["DAYS_BIRTH"])), data["CLOSED_DAYS_CREDIT_MEAN"]))
    v["i71"] = 0.099979 * np.tanh(((np.maximum(((np.where(data["BURO_DAYS_CREDIT_VAR"] < -99998, 0.318310, data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"]))), ((
        (((data["BURO_DAYS_CREDIT_VAR"]) + (data["AMT_INCOME_TOTAL"])) / 2.0))))) - ((((data["AMT_REQ_CREDIT_BUREAU_QRT"]) > (np.tanh((data["AMT_INCOME_TOTAL"])))) * 1.))))
    v["i72"] = 0.099800 * np.tanh(np.where(data["NEW_EMPLOY_TO_BIRTH_RATIO"] < -99998, ((((np.tanh(((12.86315155029296875)))) - (0.318310))) / 2.0), ((
        np.where(data["NEW_EMPLOY_TO_BIRTH_RATIO"] > 0, -2.0, ((data["NEW_EMPLOY_TO_BIRTH_RATIO"]) * (data["NEW_EMPLOY_TO_BIRTH_RATIO"])))) * 2.0)))
    v["i73"] = 0.100000 * np.tanh((((((7.0)) * 2.0)) * (((((((((data["INSTAL_DPD_MEAN"]) * ((8.18208789825439453)))) + (
        data["INSTAL_DAYS_ENTRY_PAYMENT_SUM"]))) * ((6.44760274887084961)))) + (1.570796)))))
    v["i74"] = 0.099530 * np.tanh(((np.tanh((data["CC_AMT_DRAWINGS_ATM_CURRENT_MAX"]))) + ((((np.where(data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"] > 0, (-1.0 * (
        (data["CC_AMT_DRAWINGS_ATM_CURRENT_MAX"]))), data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"])) < ((((1.0) + ((((-1.0) + (data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"])) / 2.0))) / 2.0))) * 1.))))
    v["i75"] = 0.099510 * np.tanh(((np.minimum(((((1.570796) - (np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"] > 0, data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"], data["EXT_SOURCE_1"]))))),
                                               ((((np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"] > 0, data["NAME_INCOME_TYPE_Unemployed"], data["EXT_SOURCE_1"])) - (data["NEW_SOURCES_PROD"])))))) * 2.0))
    v["i76"] = 0.098997 * np.tanh(np.where(data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"] < -99998, ((-1.0) / 2.0), np.where(data["CLOSED_DAYS_CREDIT_MEAN"] > 0, ((
        data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) * 2.0), np.where(data["NEW_SOURCES_PROD"] < -99998, ((data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]) * 2.0), ((2.0) * 2.0)))))
    v["i77"] = 0.099680 * np.tanh(np.tanh((np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, np.where(data["REFUSED_APP_CREDIT_PERC_MAX"] < -99998, np.where(
        data["NEW_SCORES_STD"] < 0, 1.0, data["NEW_SCORES_STD"]), data["REFUSED_AMT_GOODS_PRICE_MAX"]), (-1.0 * ((data["NEW_SCORES_STD"])))))))
    v["i78"] = 0.099750 * np.tanh(np.where(data["NEW_EMPLOY_TO_BIRTH_RATIO"] < -99998, 0.318310, (((-1.0 * ((((((data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]) * 2.0)) * 2.0))))) - (
        ((data["NEW_EMPLOY_TO_BIRTH_RATIO"]) + ((((data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]) > (np.tanh((data["NEW_EMPLOY_TO_BIRTH_RATIO"])))) * 1.)))))))
    v["i79"] = 0.099996 * np.tanh(((np.where(data["EXT_SOURCE_3"] < -99998, ((((data["ORGANIZATION_TYPE_Legal_Services"]) * 2.0)) * 2.0),
                                             (-1.0 * (((((((((data["EXT_SOURCE_3"]) * 2.0)) + (1.570796)) / 2.0)) + (((data["EXT_SOURCE_3"]) * 2.0)))))))) * 2.0))
    v["i80"] = 0.099060 * np.tanh(np.where(((data["ACTIVE_DAYS_CREDIT_VAR"]) + (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])) < -99998, 0.318310, ((
        ((((((((((data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]) - (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]))) * 2.0)) * 2.0)) - (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]))) * 2.0)) * 2.0)))
    v["i81"] = 0.099949 * np.tanh(np.where(data["REG_CITY_NOT_LIVE_CITY"] > 0, (-1.0 * ((data["DAYS_BIRTH"]))), (((((data["REG_CITY_NOT_WORK_CITY"]) < (
        ((np.minimum(((data["NAME_INCOME_TYPE_Maternity_leave"])), ((((data["DAYS_BIRTH"]) - (data["DAYS_REGISTRATION"])))))) - (data["DAYS_REGISTRATION"])))) * 1.)) / 2.0)))
    v["i82"] = 0.099796 * np.tanh(np.where(data["EXT_SOURCE_3"] < -99998, ((-1.0) / 2.0), ((((((np.minimum(((data["NEW_SOURCES_PROD"])), ((
        data["NEW_CREDIT_TO_INCOME_RATIO"])))) * (data["EXT_SOURCE_3"]))) + (data["NEW_CREDIT_TO_INCOME_RATIO"]))) + (np.tanh((-1.0))))))
    v["i83"] = 0.099958 * np.tanh(np.where(((data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]) * 2.0) > 0, -1.0, np.where(np.minimum(((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])), ((data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"])))
                                                                                                                 < 0, (((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) < (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"])) * 1.), (-1.0 * ((((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) * 2.0)))))))
    v["i84"] = 0.099960 * np.tanh(np.where(data["ACTIVE_CREDIT_DAY_OVERDUE_MEAN"] < 0, ((np.tanh((np.where(data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]
                                                                                                           < -99998, 1.0, ((data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]) + (np.where(data["ACTIVE_AMT_CREDIT_SUM_SUM"] > 0, (9.0), -2.0))))))) / 2.0), (8.0)))
    v["i85"] = 0.099980 * np.tanh(((((((np.tanh((-1.0))) - (data["NEW_SOURCES_PROD"]))) - (data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"])))
                                   * (((-2.0) - (((np.where(data["EXT_SOURCE_3"] < -99998, -1.0, data["EXT_SOURCE_3"])) * 2.0))))))
    v["i86"] = 0.099797 * np.tanh(np.where(data["EXT_SOURCE_1"] < -99998, ((np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < 0, data["DAYS_BIRTH"], data[
                                  "ACTIVE_DAYS_CREDIT_VAR"])) * 2.0), (-1.0 * ((((((((data["EXT_SOURCE_1"]) + (data["DAYS_BIRTH"]))) * 2.0)) + (data["DAYS_BIRTH"])))))))
    v["i87"] = 0.099960 * np.tanh(np.where(((data["NEW_EMPLOY_TO_BIRTH_RATIO"]) + (((0.0) - (data["DAYS_BIRTH"])))) < -99998, data[
                                  "DAYS_BIRTH"], np.maximum(((data["ORGANIZATION_TYPE_XNA"])), ((((((-1.0) + (data["DAYS_BIRTH"]))) * (data["NEW_EMPLOY_TO_BIRTH_RATIO"])))))))
    v["i88"] = 0.099450 * np.tanh((-1.0 * (((((np.where(data["REFUSED_DAYS_DECISION_MEAN"] < 0, ((data["REFUSED_DAYS_DECISION_MAX"]) / 2.0), data[
                                  "REFUSED_DAYS_DECISION_MEAN"])) > (np.tanh(((((data["REFUSED_AMT_CREDIT_MEAN"]) > ((-1.0 * ((data["REFUSED_DAYS_DECISION_MEAN"]))))) * 1.))))) * 1.)))))
    v["i89"] = 0.099620 * np.tanh(np.where(data["POS_SK_DPD_DEF_MAX"] > 0, (-1.0 * ((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]))), np.where(data["EXT_SOURCE_3"] < -
                                                                                                                                             99998, np.minimum(((data["POS_SK_DPD_DEF_MAX"])), ((-1.0))), np.where(data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"] < 0, (-1.0 * ((data["EXT_SOURCE_3"]))), -1.0))))
    v["i90"] = 0.099901 * np.tanh(np.tanh(((((-1.0 * ((np.maximum(((data["ACTIVE_DAYS_CREDIT_VAR"])), ((np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])), ((
        data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])))))))))) + (np.tanh((((((data["ACTIVE_DAYS_CREDIT_VAR"]) + (3.0))) * (data["ACTIVE_AMT_CREDIT_SUM_SUM"])))))))))
    v["i91"] = 0.096800 * np.tanh((((np.where(data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"] > 0, np.tanh((-1.0)), np.tanh((data["CLOSED_DAYS_CREDIT_MEAN"])))) + (np.minimum(
        (((((data["CLOSED_DAYS_CREDIT_MEAN"]) < (0.318310)) * 1.))), (((((data["CLOSED_AMT_CREDIT_SUM_SUM"]) < (data["CLOSED_CREDIT_DAY_OVERDUE_MEAN"])) * 1.)))))) / 2.0))
    v["i92"] = 0.098000 * np.tanh(np.where(data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"] < 0, (((((((1.570796) / 2.0)) * (data["DAYS_BIRTH"]))) < (
        ((3.141593) * (data["OCCUPATION_TYPE_Core_staff"])))) * 1.), np.minimum((((-1.0 * ((data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"]))))), ((data["DAYS_BIRTH"])))))
    v["i93"] = 0.098996 * np.tanh(((np.where(np.where(data["EXT_SOURCE_3"] < -99998, ((-1.0) + (data["NEW_SCORES_STD"])), (-1.0 * (
        (data["ORGANIZATION_TYPE_Bank"])))) > 0, data["EXT_SOURCE_3"], data["ORGANIZATION_TYPE_Bank"])) * (((-1.0) + (data["NEW_SCORES_STD"])))))
    v["i94"] = 0.099400 * np.tanh(np.tanh((np.maximum(((np.where(data["NAME_EDUCATION_TYPE_Higher_education"] < 0, ((data["NEW_SCORES_STD"]) - (2.0)), data["NEW_SCORES_STD"]))), ((
        ((np.where(data["NAME_EDUCATION_TYPE_Higher_education"] < 0, data["NAME_INCOME_TYPE_State_servant"], data["CLOSED_DAYS_CREDIT_VAR"])) - (data["NEW_SCORES_STD"]))))))))
    v["i95"] = 0.084695 * np.tanh(np.where(data["REGION_RATING_CLIENT_W_CITY"] > 0, data["ORGANIZATION_TYPE_Self_employed"], np.maximum(((np.where((((data["WALLSMATERIAL_MODE_Stone__brick"]) + (data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"])) / 2.0)
                                                                                                                                                   > 0, data["ORGANIZATION_TYPE_Self_employed"], data["REGION_RATING_CLIENT_W_CITY"]))), (((((((data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"]) + (data["WALLSMATERIAL_MODE_Stone__brick"])) / 2.0)) / 2.0))))))
    v["i96"] = 0.096899 * np.tanh(np.where(data["DAYS_BIRTH"] > 0, (((-1.0 * (((((3.0) < (((data["DAYS_BIRTH"]) * 2.0))) * 1.)))))
                                                                    * ((4.0))), np.where(data["REFUSED_DAYS_DECISION_MEAN"] < 0, 0.318310, ((data["DAYS_BIRTH"]) * 2.0))))
    v["i97"] = 0.097551 * np.tanh(((np.where(data["NEW_SOURCES_PROD"] > 0, data["NAME_CONTRACT_TYPE_Cash_loans"], (((((-2.0) + ((((((data["NAME_INCOME_TYPE_Unemployed"]) > (
        data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"])) * 1.)) * 2.0)))) + ((((data["NAME_CONTRACT_TYPE_Cash_loans"]) < (data["ORGANIZATION_TYPE_Legal_Services"])) * 1.))) / 2.0))) * 2.0))
    v["i98"] = 0.097900 * np.tanh((((((data["ORGANIZATION_TYPE_Legal_Services"]) + ((((data["NAME_INCOME_TYPE_Maternity_leave"]) + (((((np.maximum(((data["NAME_HOUSING_TYPE_Municipal_apartment"])), ((
        data["NAME_INCOME_TYPE_Unemployed"])))) - (((data["ORGANIZATION_TYPE_Industry__type_12"]) - (data["ORGANIZATION_TYPE_Business_Entity_Type_2"]))))) / 2.0))) / 2.0))) / 2.0)) - (data["ORGANIZATION_TYPE_Trade__type_6"])))
    v["i99"] = 0.059000 * np.tanh(np.maximum(((np.tanh((np.tanh((data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"])))))), ((((data["POS_NAME_CONTRACT_STATUS_Demand_MEAN"]) - (
        (-1.0 * (((((np.maximum(((np.tanh((data["POS_SK_DPD_MEAN"])))), ((data["POS_NAME_CONTRACT_STATUS_Demand_MEAN"])))) < (((data["POS_SK_DPD_MEAN"]) / 2.0))) * 1.))))))))))
    v["i100"] = 0.096493 * np.tanh(((((((data["REGION_POPULATION_RELATIVE"]) * (((((((np.tanh((data["REGION_POPULATION_RELATIVE"]))) < (0.636620)) * 1.)) + (
        np.tanh((data["NEW_CREDIT_TO_INCOME_RATIO"])))) / 2.0)))) - (data["FONDKAPREMONT_MODE_reg_oper_spec_account"]))) - (data["NAME_EDUCATION_TYPE_Academic_degree"])))
    v["i101"] = 0.074296 * np.tanh(np.where((((((((((data["NEW_CREDIT_TO_INCOME_RATIO"]) / 2.0)) + (data["NAME_INCOME_TYPE_Unemployed"]))) + (-2.0))) + (
        data["ORGANIZATION_TYPE_Realtor"])) / 2.0) > 0, 3.141593, ((data["OCCUPATION_TYPE_Secretaries"]) - (data["NAME_EDUCATION_TYPE_Academic_degree"]))))
    v["i102"] = 0.081810 * np.tanh((((((((data["WALLSMATERIAL_MODE_Others"]) * 2.0)) + ((((((np.maximum(((((data["NAME_INCOME_TYPE_Maternity_leave"]) + (data["NAME_TYPE_SUITE_Children"])))), ((
        data["ORGANIZATION_TYPE_Industry__type_1"])))) - (data["ORGANIZATION_TYPE_Industry__type_5"]))) + (data["ORGANIZATION_TYPE_Restaurant"])) / 2.0))) / 2.0)) - (data["ORGANIZATION_TYPE_Transport__type_1"])))
    v["i103"] = 0.098600 * np.tanh(np.where(data["DEF_30_CNT_SOCIAL_CIRCLE"] < -99998, data["DEF_30_CNT_SOCIAL_CIRCLE"], ((data["DEF_30_CNT_SOCIAL_CIRCLE"]) - (((np.where(
        data["DEF_30_CNT_SOCIAL_CIRCLE"] > 0, -2.0, ((data["ORGANIZATION_TYPE_Agriculture"]) - (data["WEEKDAY_APPR_PROCESS_START_SUNDAY"])))) * (data["DEF_30_CNT_SOCIAL_CIRCLE"]))))))
    v["i104"] = 0.091970 * np.tanh(((((((data["FLAG_DOCUMENT_3"]) * (np.maximum(((data["AMT_INCOME_TOTAL"])), (((-1.0 * (((((data["NEW_CREDIT_TO_INCOME_RATIO"]) > (
        (((data["AMT_INCOME_TOTAL"]) > (data["FLAG_DOCUMENT_3"])) * 1.))) * 1.)))))))))) - (data["ORGANIZATION_TYPE_Hotel"]))) - (data["ORGANIZATION_TYPE_Hotel"])))
    v["i105"] = 0.099940 * np.tanh(((np.where(data["EXT_SOURCE_2"] < 0, data["FONDKAPREMONT_MODE_reg_oper_spec_account"], ((
        ((((((data["FONDKAPREMONT_MODE_reg_oper_spec_account"]) + (1.0)) / 2.0)) + (data["NEW_CREDIT_TO_INCOME_RATIO"])) / 2.0)) - (((data["EXT_SOURCE_2"]) * 2.0))))) - (((data["EXT_SOURCE_2"]) * 2.0))))
    v["i106"] = 0.096000 * np.tanh(np.where((((data["NEW_INC_PER_CHLD"]) + ((((data["REGION_POPULATION_RELATIVE"]) + (data["NEW_INC_PER_CHLD"])) / 2.0))) / 2.0) < 0, ((np.where(data["CLOSED_DAYS_CREDIT_VAR"]
                                                                                                                                                                                 < -99998, data["NAME_EDUCATION_TYPE_Higher_education"], data["REGION_POPULATION_RELATIVE"])) / 2.0), (((data["DAYS_REGISTRATION"]) < (data["NAME_INCOME_TYPE_Maternity_leave"])) * 1.)))
    v["i107"] = 0.092699 * np.tanh((((((data["REGION_RATING_CLIENT"]) < ((((((data["NEW_SOURCES_PROD"]) < (-1.0)) * 1.)) - ((-1.0 * (
        (((np.minimum(((((data["REGION_RATING_CLIENT"]) * (data["NEW_SOURCES_PROD"])))), ((data["NEW_SOURCES_PROD"])))) / 2.0)))))))) * 1.)) * 2.0))
    v["i108"] = 0.099960 * np.tanh(np.minimum(((((((data["DAYS_BIRTH"]) * (((data["REGION_RATING_CLIENT_W_CITY"]) - (0.318310))))) - (np.where(np.minimum(
        ((data["DAYS_BIRTH"])), ((data["REGION_RATING_CLIENT_W_CITY"]))) < 0, data["REGION_RATING_CLIENT_W_CITY"], 0.636620))))), (((1.00496315956115723)))))
    v["i109"] = 0.089999 * np.tanh((((data["ORGANIZATION_TYPE_Mobile"]) + ((((np.minimum((((-1.0 * ((data["ORGANIZATION_TYPE_Industry__type_9"]))))), ((data["ORGANIZATION_TYPE_Other"])))) + (
        (((((data["ORGANIZATION_TYPE_Other"]) / 2.0)) + (np.minimum((((-1.0 * ((data["ORGANIZATION_TYPE_Industry__type_9"]))))), ((data["FLAG_EMP_PHONE"]))))) / 2.0))) / 2.0))) / 2.0))
    v["i110"] = 0.099760 * np.tanh(np.where((((0.318310) < (((((data["AMT_ANNUITY"]) * 2.0)) * 2.0))) * 1.) > 0, ((
        (data["REGION_POPULATION_RELATIVE"]) < (data["AMT_ANNUITY"])) * 1.), ((-1.0) + (((data["AMT_ANNUITY"]) * (data["AMT_ANNUITY"]))))))
    v["i111"] = 0.100000 * np.tanh((-1.0 * ((np.where((-1.0 * (((((data["AMT_CREDIT"]) > ((((3.141593) + (0.636620)) / 2.0))) * 1.))))
                                                      < 0, (6.10089683532714844), ((((np.maximum(((data["ACTIVE_MONTHS_BALANCE_MIN_MIN"])), ((data["AMT_CREDIT"])))) / 2.0)) / 2.0))))))
    v["i112"] = 0.094904 * np.tanh(np.where(((data["NEW_ANNUITY_TO_INCOME_RATIO"]) - (data["NEW_CREDIT_TO_INCOME_RATIO"])) < 0, ((-1.0) - (np.where(
        data["NEW_CREDIT_TO_INCOME_RATIO"] < 0, data["NEW_CREDIT_TO_INCOME_RATIO"], data["NAME_EDUCATION_TYPE_Academic_degree"]))), ((((data["NEW_CREDIT_TO_INCOME_RATIO"]) * 2.0)) + (1.0))))
    v["i113"] = 0.099204 * np.tanh((((((np.minimum(((data["AMT_CREDIT"])), (((((1.63399493694305420)) / 2.0))))) > (
        ((np.maximum(((data["BURO_DAYS_CREDIT_VAR"])), ((1.570796)))) / 2.0))) * 1.)) * (((data["NEW_CREDIT_TO_INCOME_RATIO"]) * 2.0))))
    v["i114"] = 0.004658 * np.tanh(np.where(data["REG_CITY_NOT_WORK_CITY"] > 0, (-1.0 * (((((data["BURO_DAYS_CREDIT_VAR"]) > (data["NAME_EDUCATION_TYPE_Academic_degree"])) * 1.)))), ((
        ((-1.0 * ((data["NAME_EDUCATION_TYPE_Academic_degree"])))) + ((((data["ORGANIZATION_TYPE_Legal_Services"]) + ((((data["BURO_DAYS_CREDIT_VAR"]) > (data["REG_CITY_NOT_WORK_CITY"])) * 1.))) / 2.0))) / 2.0)))
    v["i115"] = 0.090201 * np.tanh(((((((((((np.where(data["CLOSED_DAYS_CREDIT_VAR"] < -99998, data["ORGANIZATION_TYPE_Transport__type_1"], np.where(data["CLOSED_DAYS_CREDIT_VAR"]
                                                                                                                                                     < 0, data["CLOSED_DAYS_CREDIT_VAR"], data["REG_CITY_NOT_WORK_CITY"]))) * (data["NAME_EDUCATION_TYPE_Academic_degree"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0))
    v["i116"] = 0.083965 * np.tanh(np.where(data["BURO_MONTHS_BALANCE_MAX_MAX"] < 0, data["NAME_INCOME_TYPE_Maternity_leave"], np.where(data["BURO_STATUS_2_MEAN_MEAN"] > 0, ((
        ((data["BURO_MONTHS_BALANCE_MIN_MIN"]) * 2.0)) + (data["BURO_STATUS_4_MEAN_MEAN"])), ((np.minimum(((0.318310)), ((data["BURO_MONTHS_BALANCE_MIN_MIN"])))) - (data["BURO_MONTHS_BALANCE_MIN_MIN"])))))
    v["i117"] = 0.096950 * np.tanh(np.minimum((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * ((10.0)))) < (
        ((((((10.0)) > ((7.85062408447265625))) * 1.)) / 2.0))) * 1.))), (((((10.0)) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * ((8.0)))))))))
    v["i118"] = 0.099808 * np.tanh(np.minimum(((np.minimum(((data["AMT_INCOME_TOTAL"])), (((-1.0 * ((np.minimum(((data["NAME_CONTRACT_TYPE_Cash_loans"])), ((data["AMT_INCOME_TOTAL"]))))))))))), ((
        (((((((data["NAME_CONTRACT_TYPE_Revolving_loans"]) * (data["NAME_CONTRACT_TYPE_Cash_loans"]))) * (data["AMT_INCOME_TOTAL"]))) + (data["NAME_CONTRACT_TYPE_Cash_loans"])) / 2.0)))))
    v["i119"] = 0.093920 * np.tanh(((np.minimum(((data["EXT_SOURCE_1"])), ((np.where(np.where(data["ORGANIZATION_TYPE_Legal_Services"]
                                                                                              > 0, 0.636620, data["EXT_SOURCE_1"]) > 0, 0.636620, (-1.0 * ((1.570796)))))))) - (data["EXT_SOURCE_1"])))
    v["i120"] = 0.090007 * np.tanh((((0.318310) + (((((data["ORGANIZATION_TYPE_Legal_Services"]) + ((((-1.0 * ((np.maximum(((((data["NAME_INCOME_TYPE_Student"]) - (data["ORGANIZATION_TYPE_Industry__type_12"])))),
                                                                                                                           ((data["ORGANIZATION_TYPE_Industry__type_12"]))))))) - (data["NAME_EDUCATION_TYPE_Academic_degree"]))))) + (data["NAME_INCOME_TYPE_Unemployed"])))) / 2.0))
    v["i121"] = 0.096997 * np.tanh(np.minimum((((((data["EXT_SOURCE_1"]) < (1.0)) * 1.))), ((np.where(data["EXT_SOURCE_1"] > 0, data["NEW_INC_PER_CHLD"], ((
        (1.570796) < (np.where(data["EXT_SOURCE_1"] < -99998, data["NAME_INCOME_TYPE_Unemployed"], (-1.0 * ((data["EXT_SOURCE_1"])))))) * 1.))))))
    v["i122"] = 0.094855 * np.tanh(((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))) * (
        data["CODE_GENDER"]))) + ((((0.636620) < (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) * 1.))))
    v["i123"] = 0.100000 * np.tanh(((((np.where(data["EXT_SOURCE_3"] < -99998, data["NAME_INCOME_TYPE_Maternity_leave"], ((
        (((-1.0) - (data["EXT_SOURCE_3"]))) > (0.636620)) * 1.))) * 2.0)) + ((-1.0 * (((((1.0) < (data["EXT_SOURCE_3"])) * 1.)))))))
    v["i124"] = 0.097490 * np.tanh((-1.0 * ((np.where((-1.0 * (((((data["REFUSED_AMT_GOODS_PRICE_MEAN"]) > ((-1.0 * (
        ((((-1.0) > (((data["REFUSED_DAYS_DECISION_MAX"]) / 2.0))) * 1.)))))) * 1.)))) < 0, data["REFUSED_DAYS_DECISION_MAX"], (0.12027385830879211))))))
    v["i125"] = 0.091002 * np.tanh((-1.0 * ((np.tanh(((((np.where(np.where(data["REFUSED_AMT_CREDIT_MEAN"] > 0, data["REFUSED_AMT_CREDIT_MAX"], data["REFUSED_AMT_APPLICATION_MEAN"]) > 0, data[
                                   "REFUSED_CNT_PAYMENT_SUM"], np.where(data["REFUSED_CNT_PAYMENT_SUM"] > 0, data["REFUSED_AMT_APPLICATION_MAX"], data["REFUSED_AMT_CREDIT_MEAN"]))) > (data["REFUSED_AMT_CREDIT_MAX"])) * 1.)))))))
    v["i126"] = 0.091349 * np.tanh(np.where(((data["DAYS_BIRTH"]) - (data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"])) < 0, -1.0, ((
        (data["DAYS_BIRTH"]) < ((((((data["REFUSED_AMT_CREDIT_MAX"]) > (data["ACTIVE_MONTHS_BALANCE_MIN_MIN"])) * 1.)) - (1.570796)))) * 1.)))
    v["i127"] = 0.099499 * np.tanh((((data["CLOSED_CREDIT_DAY_OVERDUE_MEAN"]) > (((np.tanh((np.where(data["CLOSED_AMT_ANNUITY_MAX"] > 0, np.maximum(
        (((-1.0 * ((data["NAME_EDUCATION_TYPE_Academic_degree"]))))), ((data["CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN"]))), data["CLOSED_AMT_ANNUITY_MEAN"])))) - (data["CLOSED_AMT_ANNUITY_MAX"])))) * 1.))
    v["i128"] = 0.009900 * np.tanh(np.where(data["FLAG_EMAIL"] > 0, ((data["AMT_INCOME_TOTAL"]) - ((((data["NAME_EDUCATION_TYPE_Academic_degree"]) + (0.636620)) / 2.0))), ((
        data["NAME_INCOME_TYPE_Maternity_leave"]) - (np.maximum(((data["NAME_TYPE_SUITE_Group_of_people"])), (((((data["NAME_EDUCATION_TYPE_Academic_degree"]) + (data["FLAG_EMAIL"])) / 2.0))))))))
    v["i129"] = 0.096249 * np.tanh((((((data["ORGANIZATION_TYPE_Postal"]) + (((((((((((1.0)) + (data["ORGANIZATION_TYPE_Postal"]))) < (data["AMT_INCOME_TOTAL"])) * 1.)) - (
        data["ORGANIZATION_TYPE_Trade__type_4"]))) - (data["NAME_EDUCATION_TYPE_Academic_degree"])))) / 2.0)) + ((-1.0 * ((data["ORGANIZATION_TYPE_Transport__type_1"]))))))
    v["i130"] = 0.098970 * np.tanh((((((((((-1.0 * ((((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + ((((np.tanh((-1.0))) > (
        data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.))) / 2.0)) / 2.0)) < (data["ORGANIZATION_TYPE_Bank"])) * 1.))))) * 2.0)) * 2.0)) * 2.0)) * 2.0))
    v["i131"] = 0.092860 * np.tanh((((data["ORGANIZATION_TYPE_Telecom"]) + (np.where(data["NEW_CREDIT_TO_INCOME_RATIO"] < 0, (-1.0 * ((data["NEW_CREDIT_TO_INCOME_RATIO"]))), np.where(
        data["FLAG_EMAIL"] < 0, (((data["NEW_CREDIT_TO_INCOME_RATIO"]) < (data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.), (-1.0 * ((data["FLAG_EMAIL"]))))))) / 2.0))
    v["i132"] = 0.099999 * np.tanh((((((((np.maximum(((1.570796)), ((data["NEW_SCORES_STD"])))) < (data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"])) * 1.)) + (np.where(
        data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"] < 0, (((3.141593) < (data["NEW_SCORES_STD"])) * 1.), np.tanh((data["NEW_ANNUITY_TO_INCOME_RATIO"])))))) * 2.0))
    v["i133"] = 0.099840 * np.tanh((((((data["DAYS_BIRTH"]) * (data["NEW_SCORES_STD"]))) + (np.minimum(((((((data["DAYS_BIRTH"]) * 2.0)) * (
        (-1.0 * ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))))), ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))))) / 2.0))
    v["i134"] = 0.099021 * np.tanh(((np.where(((((((3.0) - (1.570796))) - (data["NAME_INCOME_TYPE_Student"]))) - (
        data["DAYS_BIRTH"])) > 0, ((data["NAME_EDUCATION_TYPE_Academic_degree"]) * (((data["NEW_SCORES_STD"]) * 2.0))), -2.0)) * 2.0))
    v["i135"] = 0.084999 * np.tanh(np.where(data["NAME_HOUSING_TYPE_With_parents"] > 0, (((data["AMT_ANNUITY"]) < (0.318310)) * 1.), ((1.0) - (
        np.maximum(((data["AMT_ANNUITY"])), ((np.maximum(((0.636620)), (((((data["AMT_ANNUITY"]) < (data["NAME_HOUSING_TYPE_With_parents"])) * 1.)))))))))))
    v["i136"] = 0.095405 * np.tanh(np.where((((((data["NAME_INCOME_TYPE_Student"]) + (np.minimum(((data["INSTAL_PAYMENT_PERC_VAR"])), ((data["INSTAL_PAYMENT_DIFF_STD"]))))))
                                              > (((data["INSTAL_PAYMENT_DIFF_VAR"]) * 2.0))) * 1.) > 0, 0.636620, np.minimum((((-1.0 * ((0.318310))))), ((data["INSTAL_AMT_PAYMENT_STD"])))))
    v["i137"] = 0.099960 * np.tanh(np.where((((((np.where(data["INSTAL_PAYMENT_PERC_MAX"] > 0, (-1.0 * ((data["NAME_INCOME_TYPE_Maternity_leave"]))), data["NAME_EDUCATION_TYPE_Academic_degree"])) / 2.0))
                                              < (data["INSTAL_PAYMENT_PERC_MEAN"])) * 1.) > 0, -2.0, (((data["INSTAL_PAYMENT_PERC_MAX"]) > (data["INSTAL_PAYMENT_PERC_MEAN"])) * 1.)))
    v["i138"] = 0.094701 * np.tanh(np.where(data["YEARS_BEGINEXPLUATATION_MODE"] < -99998, (-1.0 * ((0.318310))), (-1.0 * ((np.where(
        data["YEARS_BEGINEXPLUATATION_MODE"] < 0, data["LIVINGAREA_AVG"], ((data["YEARS_BEGINEXPLUATATION_MODE"]) + (((data["YEARS_BEGINEXPLUATATION_MODE"]) * 2.0)))))))))
    v["i139"] = 0.099971 * np.tanh(((((((np.where(data["ORGANIZATION_TYPE_Realtor"] < 0, np.minimum(((data["NAME_INCOME_TYPE_Maternity_leave"])), ((
        ((2.0) - (data["AMT_ANNUITY"]))))), 2.0)) - (data["NAME_INCOME_TYPE_Student"]))) - (data["NAME_EDUCATION_TYPE_Academic_degree"]))) * 2.0))
    v["i140"] = 0.099589 * np.tanh((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) > (0.318310)) * 1.)) - (
        (((data["NAME_EDUCATION_TYPE_Academic_degree"]) > (((data["AMT_INCOME_TOTAL"]) * (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0))))) * 1.))))
    v["i141"] = 0.061000 * np.tanh(((((-1.0 * (((((data["OCCUPATION_TYPE_Drivers"]) < (data["REFUSED_AMT_ANNUITY_MEAN"])) * 1.))))) + (np.where(data["REFUSED_CNT_PAYMENT_MEAN"] < -99998, np.tanh(
        (data["NAME_EDUCATION_TYPE_Higher_education"])), np.minimum((((-1.0 * ((data["NAME_EDUCATION_TYPE_Higher_education"]))))), ((data["REFUSED_CNT_PAYMENT_MEAN"])))))) / 2.0))
    v["i142"] = 0.029650 * np.tanh(((np.minimum(((((((((data["NAME_INCOME_TYPE_Student"]) * (data["NAME_INCOME_TYPE_Student"]))) + (data["AMT_INCOME_TOTAL"]))) * (
        data["NAME_TYPE_SUITE_Group_of_people"])))), ((((data["NAME_TYPE_SUITE_Group_of_people"]) * (((data["NAME_INCOME_TYPE_Working"]) + (data["OCCUPATION_TYPE_Drivers"])))))))) * 2.0))
    v["i143"] = 0.096011 * np.tanh(np.minimum(((((0.318310) * (0.318310)))), ((((np.minimum(((((((data["AMT_ANNUITY"]) * 2.0)) * 2.0))), ((
        0.636620)))) * (np.where(data["NAME_INCOME_TYPE_Maternity_leave"] < 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], data["AMT_ANNUITY"])))))))
    v["i144"] = 0.074027 * np.tanh(((np.tanh(((((data["DAYS_EMPLOYED"]) < ((((-1.0 * ((np.maximum(((data["NEW_EMPLOY_TO_BIRTH_RATIO"])),
                                                                                                  ((((data["NAME_INCOME_TYPE_Pensioner"]) - ((-1.0 * ((0.318310)))))))))))) / 2.0))) * 1.)))) / 2.0))
    v["i145"] = 0.098999 * np.tanh(((((((((((1.570796) - ((-1.0 * ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))) > (
        0.318310)) * 1.)) / 2.0)) > (((1.570796) - ((-1.0 * ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))))) * 1.))
    v["i146"] = 0.064730 * np.tanh((-1.0 * (((((((data["BURO_MONTHS_BALANCE_MAX_MAX"]) > (data["NAME_INCOME_TYPE_Maternity_leave"])) * 1.)) * (
        (((((((0.0) > (data["BURO_STATUS_0_MEAN_MEAN"])) * 1.)) + ((-1.0 * ((data["BURO_MONTHS_BALANCE_MAX_MAX"])))))) + (data["BURO_STATUS_0_MEAN_MEAN"]))))))))
    v["i147"] = 0.099995 * np.tanh((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < ((((0.13536217808723450)) / 2.0))) * 1.)) * (
        ((((((((0.636620) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) / 2.0)) > ((0.13536217808723450))) * 1.)) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0))))))
    v["i148"] = 0.099000 * np.tanh(np.minimum((((((data["NAME_CONTRACT_TYPE_Revolving_loans"]) < (data["EXT_SOURCE_3"])) * 1.))), ((
        ((((np.where(data["NEW_SOURCES_PROD"] > 0, data["NEW_DOC_IND_KURT"], (((((1.0) + (data["NEW_DOC_IND_KURT"])) / 2.0)) - (data["EXT_SOURCE_3"])))) * 2.0)) * 2.0)))))
    v["i149"] = 0.099300 * np.tanh(np.minimum((((((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) > (data["ORGANIZATION_TYPE_Industry__type_2"])) * 1.))), ((((np.minimum(((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])), ((
        (-1.0 * ((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]))))))) - (np.where(data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"] < 0, data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"], ((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) * 2.0))))))))
    v["i150"] = 0.020000 * np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"] < -99998, data["AMT_INCOME_TOTAL"], np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, 0.0, np.where(
        np.maximum(((data["ACTIVE_DAYS_CREDIT_VAR"])), ((data["AMT_INCOME_TOTAL"]))) > 0, data["ACTIVE_DAYS_CREDIT_VAR"], (-1.0 * ((((data["NAME_INCOME_TYPE_Working"]) / 2.0))))))))
    v["i151"] = 0.099731 * np.tanh(np.where((((((data["NAME_EDUCATION_TYPE_Lower_secondary"]) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) < 0, ((
        (((((data["NAME_EDUCATION_TYPE_Lower_secondary"]) / 2.0)) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)) + (data["NAME_EDUCATION_TYPE_Lower_secondary"])), ((data["NAME_EDUCATION_TYPE_Lower_secondary"]) * 2.0)))
    v["i152"] = 0.034897 * np.tanh(np.where(data["NEW_DOC_IND_KURT"] > 0, ((np.where((-1.0 * ((data["EXT_SOURCE_3"]))) > 0, (-1.0 * ((0.636620))), data[
                                   "AMT_CREDIT"])) / 2.0), np.where(data["NAME_INCOME_TYPE_Student"] < 0, (-1.0 * ((data["EXT_SOURCE_3"]))), data["NEW_DOC_IND_KURT"])))
    v["i153"] = 0.088029 * np.tanh(((((((1.570796) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)) < (
        (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (((1.570796) + (-1.0)))) * 1.))) * 1.))
    v["i154"] = 0.097020 * np.tanh(((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * ((((((data["ACTIVE_DAYS_CREDIT_VAR"]) > (data["AMT_INCOME_TOTAL"]))
                                                                                * 1.)) - (((((((data["AMT_ANNUITY"]) < (0.318310)) * 1.)) < (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])) * 1.))))))
    v["i155"] = 0.060999 * np.tanh((-1.0 * (((((((((((((data["NAME_INCOME_TYPE_Student"]) - (
        3.0))) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) - (3.141593))) > (1.0)) * 1.)) * 2.0)))))
    v["i156"] = 0.099598 * np.tanh(((((data["AMT_INCOME_TOTAL"]) - ((((np.tanh((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) / 2.0)))) < (
        data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])) * 1.)))) * ((((np.tanh(((-1.0 * ((data["AMT_INCOME_TOTAL"])))))) < (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])) * 1.))))
    v["i157"] = 0.099740 * np.tanh((((((((3.0) < (((((np.where(data["AMT_CREDIT"] < 0, (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (
        (-1.0 * ((0.636620))))) * 1.), 0.636620)) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) * 2.0))) * 1.)) * 2.0)) * 2.0))
    v["i158"] = 0.098100 * np.tanh(((((np.minimum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((((np.where(data["NAME_INCOME_TYPE_Working"] > 0, (-1.0 * (
        (((data["NEW_DOC_IND_KURT"]) * (data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]))))), data["NEW_DOC_IND_KURT"])) - (data["NAME_INCOME_TYPE_Working"])))))) * (data["FLAG_EMP_PHONE"]))) / 2.0))
    v["i159"] = 0.099450 * np.tanh(((((data["NEW_DOC_IND_KURT"]) / 2.0)) * (np.where(((0.318310) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))
                                                                                     > 0, np.where(data["EXT_SOURCE_3"] < 0, ((data["AMT_ANNUITY"]) / 2.0), data["EXT_SOURCE_3"]), data["EXT_SOURCE_3"]))))
    v["i160"] = 0.099801 * np.tanh(np.where(data["ACTIVE_DAYS_CREDIT_VAR"] > 0, np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"] < 0, 0.0, ((
        (4.0)) - (data["ACTIVE_DAYS_CREDIT_VAR"]))), (((data["ACTIVE_DAYS_CREDIT_VAR"]) > (((1.0) + (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])))) * 1.)))
    v["i161"] = 0.076150 * np.tanh((((((((data["CLOSED_MONTHS_BALANCE_MIN_MIN"]) > (np.where(data["CLOSED_MONTHS_BALANCE_MAX_MAX"] < 0, data["NAME_INCOME_TYPE_Maternity_leave"], data["ACTIVE_DAYS_CREDIT_VAR"]))) * 1.)) / 2.0)) - (
        (((data["CLOSED_MONTHS_BALANCE_MAX_MAX"]) > (np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < 0, data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"], data["NAME_INCOME_TYPE_Maternity_leave"]))) * 1.))))
    v["i162"] = 0.099900 * np.tanh(((((data["EXT_SOURCE_2"]) * (((((np.tanh((np.tanh((np.minimum(((((data["EXT_SOURCE_2"]) * 2.0))),
                                                                                                 ((data["AMT_ANNUITY"])))))))) - (data["EXT_SOURCE_2"]))) * 2.0)))) * (((data["EXT_SOURCE_2"]) * 2.0))))
    v["i163"] = 0.081001 * np.tanh((((np.minimum(((data["OCCUPATION_TYPE_High_skill_tech_staff"])), ((data["HOUR_APPR_PROCESS_START"])))) > (np.where(data["AMT_INCOME_TOTAL"] < 0, np.minimum(
        ((data["REGION_RATING_CLIENT_W_CITY"])), ((np.where(data["REGION_RATING_CLIENT"] < 0, ((data["AMT_INCOME_TOTAL"]) / 2.0), data["NAME_INCOME_TYPE_Maternity_leave"])))), data["AMT_INCOME_TOTAL"]))) * 1.))
    v["i164"] = 0.099752 * np.tanh(((np.where(data["CLOSED_DAYS_CREDIT_VAR"] < 0, np.minimum(((np.where(data["AMT_INCOME_TOTAL"] < 0, ((data["REGION_RATING_CLIENT_W_CITY"]) - (data["WALLSMATERIAL_MODE_Panel"])),
                                                                                                        (-1.0 * ((data["REGION_RATING_CLIENT_W_CITY"])))))), ((data["NAME_INCOME_TYPE_Maternity_leave"]))), data["WALLSMATERIAL_MODE_Panel"])) * (data["REGION_RATING_CLIENT_W_CITY"])))
    v["i165"] = 0.098940 * np.tanh(np.maximum(((data["CLOSED_CREDIT_DAY_OVERDUE_MEAN"])), ((np.where(data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"] < 0, np.where(data["CLOSED_DAYS_CREDIT_VAR"] < 0, np.tanh(
        (data["AMT_INCOME_TOTAL"])), data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]), np.where(np.tanh((data["AMT_INCOME_TOTAL"])) < 0, data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"], data["CLOSED_DAYS_CREDIT_VAR"]))))))
    v["i166"] = 0.083961 * np.tanh((((3.141593) < ((((-1.0 * ((np.minimum(((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] > 0, data["DAYS_BIRTH"], np.where(
        data["DAYS_BIRTH"] > 0, data["DAYS_BIRTH"], data["NEW_CREDIT_TO_ANNUITY_RATIO"])))), ((data["ORGANIZATION_TYPE_Advertising"]))))))) * 2.0))) * 1.))
    v["i167"] = 0.099970 * np.tanh(((((data["OCCUPATION_TYPE_Low_skill_Laborers"]) * (((data["NAME_FAMILY_STATUS_Married"]) * (((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (
        ((data["ORGANIZATION_TYPE_Advertising"]) + (data["OCCUPATION_TYPE_Low_skill_Laborers"]))))) + ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["NEW_ANNUITY_TO_INCOME_RATIO"])) / 2.0)))))))) * 2.0))
    v["i168"] = 0.069500 * np.tanh(((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (np.maximum(((np.tanh(((((4.77533054351806641)) * (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (
        data["NAME_TYPE_SUITE_Group_of_people"])))))))), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))) - (data["OCCUPATION_TYPE_Low_skill_Laborers"])))
    v["i169"] = 0.098609 * np.tanh(np.minimum(((np.maximum((((((data["ACTIVE_DAYS_CREDIT_VAR"]) < ((((((-2.0) / 2.0)) < (data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.))) * 1.))),
                                                           ((data["AMT_ANNUITY"]))))), ((((((((0.318310) < (data["AMT_ANNUITY"])) * 1.)) < (data["ACTIVE_DAYS_CREDIT_VAR"])) * 1.)))))
    v["i170"] = 0.099983 * np.tanh((((((-1.0 * ((np.where(data["BURO_CREDIT_TYPE_Mortgage_MEAN"] > 0, 3.0, np.where(data["ORGANIZATION_TYPE_Advertising"] < 0, data[
                                   "ORGANIZATION_TYPE_Industry__type_2"], ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["ACTIVE_DAYS_CREDIT_VAR"]))) - (3.0)))))))) * 2.0)) * 2.0))
    v["i171"] = 0.096960 * np.tanh(np.where(((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (data["NAME_EDUCATION_TYPE_Higher_education"])) > 0, data["OCCUPATION_TYPE_Accountants"], ((
        (((((data["NAME_EDUCATION_TYPE_Higher_education"]) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) < ((((data["OCCUPATION_TYPE_Accountants"]) < (data["NAME_TYPE_SUITE_Other_A"])) * 1.))) * 1.)) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))
    v["i172"] = 0.099790 * np.tanh(((((np.where(data["NAME_TYPE_SUITE_Other_A"] > 0, data["NEW_ANNUITY_TO_INCOME_RATIO"], ((((-1.0 * ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)))))
                                                                                                                            > (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, 3.0, ((data["NAME_TYPE_SUITE_Other_A"]) * 2.0)))) * 1.))) * 2.0)) * 2.0))
    v["i173"] = 0.093300 * np.tanh(np.where(((data["DAYS_BIRTH"]) / 2.0) < 0, (((data["NAME_TYPE_SUITE_Other_A"]) < (((data["DAYS_BIRTH"]) / 2.0)))
                                                                               * 1.), (((data["NAME_TYPE_SUITE_Other_A"]) > ((((5.23197555541992188)) - (data["ACTIVE_DAYS_CREDIT_VAR"])))) * 1.)))
    v["i174"] = 0.085000 * np.tanh((-1.0 * ((np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, data["NAME_TYPE_SUITE_Other_A"], (((np.where(
        data["ACTIVE_DAYS_CREDIT_VAR"] < 0, (-1.0 * ((0.318310))), data["NAME_TYPE_SUITE_Other_A"])) < (data["REFUSED_AMT_ANNUITY_MEAN"])) * 1.))))))
    v["i175"] = 0.099900 * np.tanh((((((data["ACTIVE_AMT_ANNUITY_MAX"]) > (np.where(np.where(data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"] < -99998, (6.76153230667114258), ((
        (data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"]) + (data["ACTIVE_MONTHS_BALANCE_MIN_MIN"])) / 2.0)) < 0, ((((data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"]) / 2.0)) / 2.0), 0.318310))) * 1.)) * 2.0))
    v["i176"] = 0.095999 * np.tanh((-1.0 * ((np.maximum(((np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] < 0, data["NAME_EDUCATION_TYPE_Lower_secondary"], 0.318310))), ((
        np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] < 0, ((((3.92553544044494629)) < (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0))) * 1.), ((0.318310) / 2.0)))))))))
    v["i177"] = 0.040399 * np.tanh(((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < -99998, data["NEW_CREDIT_TO_ANNUITY_RATIO"], ((
        (np.tanh((1.0))) < (((data["NEW_CREDIT_TO_INCOME_RATIO"]) + ((((-1.0) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.))))) * 1.))) / 2.0))
    v["i178"] = 0.094910 * np.tanh(np.where(data["DAYS_BIRTH"] < 0, (((data["NEW_CREDIT_TO_INCOME_RATIO"]) < (-1.0)) * 1.), np.minimum(
        ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (1.0)))), (((((data["DAYS_BIRTH"]) < (((-1.0) * (data["NEW_CREDIT_TO_INCOME_RATIO"])))) * 1.))))))
    v["i179"] = 0.099100 * np.tanh(np.minimum(((np.where(data["LIVE_REGION_NOT_WORK_REGION"] < 0, (((data["OCCUPATION_TYPE_Sales_staff"]) > (data["NAME_INCOME_TYPE_Maternity_leave"])) * 1.), data[
                                   "OCCUPATION_TYPE_Sales_staff"]))), ((((((np.minimum(((((data["OCCUPATION_TYPE_Sales_staff"]) * 2.0))), ((data["ACTIVE_DAYS_CREDIT_VAR"])))) / 2.0)) * (data["NAME_INCOME_TYPE_Student"]))))))
    v["i180"] = 0.095002 * np.tanh(((np.minimum(((data["NAME_INCOME_TYPE_Maternity_leave"])), ((np.where((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (
        data["NEW_CREDIT_TO_INCOME_RATIO"])) / 2.0))) / 2.0) < 0, ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)) * (data["NEW_CREDIT_TO_INCOME_RATIO"])), data["NEW_CREDIT_TO_INCOME_RATIO"]))))) * 2.0))
    v["i181"] = 0.068969 * np.tanh((((((((data["NAME_CONTRACT_TYPE_Cash_loans"]) + (((((data["DAYS_BIRTH"]) * ((((-1.0 * ((data["DAYS_BIRTH"])))) * (data["NEW_CREDIT_TO_INCOME_RATIO"])))))
                                                                                     * (data["NAME_CONTRACT_TYPE_Revolving_loans"])))) / 2.0)) + (data["ORGANIZATION_TYPE_Mobile"]))) + (data["ORGANIZATION_TYPE_Mobile"])))
    v["i182"] = 0.065961 * np.tanh(np.minimum(((np.where(data["CLOSED_DAYS_CREDIT_VAR"] > 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], np.where(data["OCCUPATION_TYPE_Sales_staff"] > 0, data[
                                   "CLOSED_DAYS_CREDIT_VAR"], ((data["NAME_EDUCATION_TYPE_Academic_degree"]) * (((data["OCCUPATION_TYPE_Sales_staff"]) * 2.0))))))), (((((0.318310) > (data["CLOSED_DAYS_CREDIT_VAR"])) * 1.)))))
    v["i183"] = 0.096998 * np.tanh(np.maximum((((((6.0)) * (((data["NAME_EDUCATION_TYPE_Academic_degree"]) * (((data["AMT_CREDIT"]) - (data["NEW_ANNUITY_TO_INCOME_RATIO"])))))))),
                                              ((np.minimum(((((data["AMT_CREDIT"]) - (data["NEW_ANNUITY_TO_INCOME_RATIO"])))), ((data["OCCUPATION_TYPE_Sales_staff"])))))))
    v["i184"] = 0.099830 * np.tanh((((((((np.maximum((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0))) / 2.0))), ((
        ((np.tanh((1.570796))) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))) + (-2.0))) > (data["ORGANIZATION_TYPE_Trade__type_4"])) * 1.)) / 2.0))
    v["i185"] = 0.094980 * np.tanh((((((data["AMT_INCOME_TOTAL"]) > (np.maximum(((np.maximum(((((data["DAYS_BIRTH"]) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))),
                                                                                             ((data["DAYS_BIRTH"]))))), (((((data["CLOSED_DAYS_CREDIT_VAR"]) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)))))) * 1.)) * 2.0))
    v["i186"] = 0.097300 * np.tanh(np.minimum(((((data["AMT_ANNUITY"]) * ((((((data["AMT_ANNUITY"]) * ((-1.0 * (
        (data["NEW_DOC_IND_KURT"])))))) + (data["DAYS_BIRTH"])) / 2.0))))), ((data["ORGANIZATION_TYPE_Trade__type_4"]))))
    v["i187"] = 0.098399 * np.tanh((-1.0 * ((np.tanh(((((np.minimum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), ((data["NAME_HOUSING_TYPE_Office_apartment"])))) + (
        (((((((data["NAME_HOUSING_TYPE_Office_apartment"]) > (((1.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) * 1.)) * 2.0)) * 2.0))) / 2.0)))))))
    v["i188"] = 0.037011 * np.tanh(((data["ORGANIZATION_TYPE_Trade__type_6"]) * (np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] < 0, ((
        ((np.where(data["AMT_ANNUITY"] < 0, data["AMT_ANNUITY"], 1.570796)) * 2.0)) * 2.0), ((1.0) - (data["AMT_ANNUITY"]))))))
    v["i189"] = 0.075050 * np.tanh((((3.141593) < ((-1.0 * (((((-1.0 * (((((-1.0) < (np.minimum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])),
                                                                                                (((-1.0 * ((data["AMT_ANNUITY"])))))))) * 1.))))) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)))))))) * 1.))
    v["i190"] = 0.097999 * np.tanh((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (data["AMT_ANNUITY"])) * 1.)) * (data["WALLSMATERIAL_MODE_Panel"]))) * (
        np.tanh((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)) - (data["AMT_INCOME_TOTAL"]))) - (data["AMT_INCOME_TOTAL"])))))))
    v["i191"] = 0.070000 * np.tanh(((((np.where(data["NONLIVINGAREA_MODE"] > 0, (((((2.0) < (data["FLOORSMAX_MODE"])) * 1.)) - (
        (((data["NONLIVINGAREA_MODE"]) < (data["NONLIVINGAREA_AVG"])) * 1.))), ((data["NONLIVINGAREA_AVG"]) - (data["NONLIVINGAREA_MODE"])))) * 2.0)) * 2.0))
    v["i192"] = 0.092500 * np.tanh(((((((((-1.0) + (data["FLOORSMAX_MEDI"])) / 2.0)) > ((0.13370278477668762))) * 1.)) - (
        (((data["FLOORSMAX_MODE"]) > ((((((data["FLOORSMAX_MODE"]) * (data["YEARS_BUILD_MEDI"]))) < (np.tanh((data["FLOORSMAX_MEDI"])))) * 1.))) * 1.))))
    v["i193"] = 0.095001 * np.tanh(((((((((((data["NONLIVINGAREA_MODE"]) + ((((data["YEARS_BUILD_MODE"]) < ((-1.0 * ((data["NONLIVINGAREA_MODE"]))))) * 1.)))) + (
        data["NEW_CREDIT_TO_INCOME_RATIO"])) / 2.0)) > (0.636620)) * 1.)) - ((((0.318310) < (data["NONLIVINGAREA_MODE"])) * 1.))))
    v["i194"] = 0.095010 * np.tanh((((((data["FLOORSMAX_MODE"]) * (((1.570796) * (data["APARTMENTS_MEDI"]))))) < (np.where(
        data["NONLIVINGAREA_MEDI"] < -99998, np.where(data["FLOORSMAX_MEDI"] > 0, 0.318310, data["APARTMENTS_AVG"]), -1.0))) * 1.))
    v["i195"] = 0.079990 * np.tanh((-1.0 * ((((((((data["APARTMENTS_AVG"]) < (np.where(data["FLOORSMAX_AVG"] < 0, np.tanh((data["ENTRANCES_MODE"])), np.where(
        data["NONLIVINGAREA_MEDI"] < -99998, data["APARTMENTS_AVG"], 0.318310)))) * 1.)) + (np.tanh((data["ENTRANCES_AVG"])))) / 2.0)))))
    v["i196"] = 0.091496 * np.tanh(((((((1.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) * ((((2.0) > ((((((3.141593) + (
        np.where(data["AMT_ANNUITY"] > 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], data["AMT_ANNUITY"]))) / 2.0)) * 2.0))) * 1.)))) * 2.0))
    v["i197"] = 0.074650 * np.tanh(((data["NAME_CONTRACT_TYPE_Cash_loans"]) * (np.maximum(((data["NAME_INCOME_TYPE_Maternity_leave"])), (((((((data["NAME_CONTRACT_TYPE_Revolving_loans"]) * (np.minimum(
        (((-1.0 * ((((data["NAME_CONTRACT_TYPE_Cash_loans"]) * (data["ORGANIZATION_TYPE_Trade__type_4"]))))))), ((data["NAME_CONTRACT_TYPE_Revolving_loans"])))))) < (data["AMT_INCOME_TOTAL"])) * 1.)))))))
    v["i198"] = 0.096000 * np.tanh(((((-1.0 * ((np.where(((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["NAME_TYPE_SUITE_Other_A"])) < 0, ((data["NAME_TYPE_SUITE_Other_A"]) / 2.0), data[
                                   "NAME_INCOME_TYPE_Student"]))))) > ((((0.636620) + ((((0.318310) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) / 2.0))) / 2.0))) * 1.))
    v["i199"] = 0.022986 * np.tanh(((((((np.where(data["AMT_ANNUITY"] > 0, 2.0, ((data["NAME_EDUCATION_TYPE_Academic_degree"]) - (
        2.0)))) - (data["AMT_ANNUITY"]))) - (data["NEW_ANNUITY_TO_INCOME_RATIO"]))) * (data["ORGANIZATION_TYPE_Trade__type_6"])))
    v["i200"] = 0.059999 * np.tanh(np.minimum(((data["ORGANIZATION_TYPE_Trade__type_4"])), ((np.where((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (1.570796)) * 1.) > 0, (-1.0 * (
        (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] > 0, data["ORGANIZATION_TYPE_Trade__type_4"], ((data["AMT_ANNUITY"]) + (-2.0)))))), data["AMT_ANNUITY"])))))
    v["i201"] = 0.087580 * np.tanh(((np.where(((data["BURO_STATUS_2_MEAN_MEAN"]) + (data["BURO_STATUS_4_MEAN_MEAN"])) < 0, (((0.318310) < (
        (((data["AMT_ANNUITY"]) + (((data["BURO_STATUS_2_MEAN_MEAN"]) + (data["BURO_STATUS_4_MEAN_MEAN"])))) / 2.0))) * 1.), -2.0)) * 2.0))
    v["i202"] = 0.099718 * np.tanh((((((-1.0 * (((((np.where(np.where(data["BURO_STATUS_0_MEAN_MEAN"] < 0, data["BURO_STATUS_X_MEAN_MEAN"], data[
                                   "BURO_STATUS_0_MEAN_MEAN"]) < 0, ((np.tanh((-1.0))) / 2.0), (2.05887126922607422))) < (data["BURO_STATUS_0_MEAN_MEAN"])) * 1.))))) * 2.0)) * 2.0))
    v["i203"] = 0.097371 * np.tanh((((((np.where(((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]) - (data["NEW_ANNUITY_TO_INCOME_RATIO"])) > 0, 2.0, ((data["NAME_EDUCATION_TYPE_Academic_degree"]) / 2.0)))
                                       < (((((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]) - (data["NAME_EDUCATION_TYPE_Academic_degree"]))) - (data["NEW_ANNUITY_TO_INCOME_RATIO"])))) * 1.)) * 2.0))
    v["i204"] = 0.094501 * np.tanh(((np.where(((np.where(data["BURO_MONTHS_BALANCE_SIZE_MEAN"] < 0, data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"], data["BURO_MONTHS_BALANCE_SIZE_MEAN"])
                                                ) + (-1.0)) > 0, -2.0, (((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]) > (np.tanh((np.tanh((data["BURO_MONTHS_BALANCE_SIZE_MEAN"])))))) * 1.))) * 2.0))
    v["i205"] = 0.096020 * np.tanh((((((((data["ACTIVE_AMT_ANNUITY_MEAN"]) > ((((data["ACTIVE_AMT_ANNUITY_MAX"]) > (data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"])) * 1.))) * 1.)) * 2.0)) - (
        np.maximum((((((data["ACTIVE_AMT_ANNUITY_MEAN"]) > (data["ACTIVE_AMT_ANNUITY_MAX"])) * 1.))), (((((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]) > (data["ACTIVE_AMT_ANNUITY_MAX"])) * 1.)))))))
    v["i206"] = 0.023997 * np.tanh(((((np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, 0.318310, (((np.tanh((np.tanh((np.tanh(
        (data["ACTIVE_DAYS_CREDIT_VAR"]))))))) > (0.318310)) * 1.))) / 2.0)) - ((((3.141593) < (data["ACTIVE_DAYS_CREDIT_VAR"])) * 1.))))
    v["i207"] = 0.098400 * np.tanh((-1.0 * ((((((((data["ORGANIZATION_TYPE_Trade__type_4"]) + (((data["NAME_INCOME_TYPE_Student"]) + (data["NAME_EDUCATION_TYPE_Academic_degree"]))))) / 2.0)) + (
        ((((data["NAME_TYPE_SUITE_Group_of_people"]) * 2.0)) + (((data["ORGANIZATION_TYPE_Trade__type_6"]) - (data["ORGANIZATION_TYPE_Industry__type_4"]))))))))))
    v["i208"] = 0.095998 * np.tanh(((data["NAME_INCOME_TYPE_Maternity_leave"]) - (np.where((((((np.maximum(((data["AMT_ANNUITY"])), ((3.141593)))) - (
        data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) > (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0))) * 1.) > 0, data["NAME_INCOME_TYPE_Student"], (5.0)))))
    v["i209"] = 0.000001 * np.tanh(np.where(((-2.0) + (data["ACTIVE_DAYS_CREDIT_VAR"])) > 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], np.where(
        data["ACTIVE_DAYS_CREDIT_VAR"] > 0, (-1.0 * (((((2.0) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)))), (((2.0) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.))))
    v["i210"] = 0.099010 * np.tanh(((((data["AMT_ANNUITY"]) - (((((np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, 0.636620, np.minimum(((np.minimum(
        ((data["ACTIVE_DAYS_CREDIT_VAR"])), ((0.0))))), ((data["NAME_EDUCATION_TYPE_Academic_degree"]))))) * 2.0)) * 2.0)))) * (data["NAME_TYPE_SUITE_Other_A"])))
    v["i211"] = 0.099733 * np.tanh(((((((((data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]) + (data["CLOSED_MONTHS_BALANCE_SIZE_SUM"])) / 2.0)) > (0.318310)) * 1.)) * (
        ((((((1.0) - (((data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"]) / 2.0)))) - (((data["CLOSED_MONTHS_BALANCE_SIZE_SUM"]) / 2.0)))) * 2.0))))
    v["i212"] = 0.096050 * np.tanh(np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"] > 0, np.where(data["CLOSED_AMT_CREDIT_SUM_SUM"] > 0, ((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])
                                                                                                                                        * 2.0), (-1.0 * (((((-1.0) > (data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"])) * 1.))))), (-1.0 * (((((data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]) > (1.0)) * 1.))))))
    v["i213"] = 0.099993 * np.tanh(np.where(data["BURO_AMT_CREDIT_SUM_MEAN"] < -99998, ((data["EXT_SOURCE_3"]) - (data["AMT_REQ_CREDIT_BUREAU_WEEK"])), ((
        ((((data["AMT_REQ_CREDIT_BUREAU_QRT"]) - (3.141593))) * (data["BURO_AMT_CREDIT_SUM_MEAN"]))) * ((((data["AMT_REQ_CREDIT_BUREAU_QRT"]) > (data["BURO_AMT_CREDIT_SUM_MEAN"])) * 1.)))))
    v["i214"] = 0.099000 * np.tanh(np.where((((-2.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) / 2.0) > 0, 1.570796, (-1.0 * (((((data["EXT_SOURCE_3"]) > (
        np.where(data["BURO_DAYS_CREDIT_ENDDATE_MEAN"] > 0, (((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]) > (1.570796)) * 1.), 1.570796))) * 1.))))))
    v["i215"] = 0.099496 * np.tanh(np.where(data["EXT_SOURCE_3"] < -99998, data["NAME_INCOME_TYPE_Maternity_leave"], np.where((((2.0) < (
        ((data["NAME_INCOME_TYPE_Maternity_leave"]) - (data["EXT_SOURCE_3"])))) * 1.) > 0, (((4.0)) * 2.0), ((data["AMT_REQ_CREDIT_BUREAU_DAY"]) + (data["AMT_REQ_CREDIT_BUREAU_DAY"])))))
    v["i216"] = 0.052589 * np.tanh(((((((((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]) - ((((data["AMT_REQ_CREDIT_BUREAU_YEAR"]) + (data["AMT_REQ_CREDIT_BUREAU_DAY"])) / 2.0))))
                                        * ((-1.0 * (((((data["EXT_SOURCE_3"]) > ((((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]) > (0.0)) * 1.))) * 1.))))))) * 2.0)) * 2.0))
    v["i217"] = 0.099605 * np.tanh(((np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"])), (((((np.where(data["ACTIVE_CNT_CREDIT_PROLONG_SUM"] > 0, data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"], ((
        ((data["ACTIVE_CNT_CREDIT_PROLONG_SUM"]) + (data["ACTIVE_CNT_CREDIT_PROLONG_SUM"]))) + (((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]) / 2.0))))) > (data["ACTIVE_AMT_CREDIT_SUM_MEAN"])) * 1.))))) * 2.0))
    v["i218"] = 0.099979 * np.tanh(np.where(data["REFUSED_DAYS_DECISION_MAX"] > 0, (-1.0 * (((((data["REFUSED_CNT_PAYMENT_SUM"]) > (
        data["REFUSED_AMT_CREDIT_MEAN"])) * 1.)))), (-1.0 * (((((data["REFUSED_CNT_PAYMENT_SUM"]) > (0.318310)) * 1.))))))
    v["i219"] = 0.089880 * np.tanh(((((((np.minimum(((data["REFUSED_CNT_PAYMENT_SUM"])), ((np.where(((data["REFUSED_DAYS_DECISION_MEAN"]) * 2.0) < 0, data["REFUSED_HOUR_APPR_PROCESS_START_MAX"], -1.0)))))
                                        > (data["REFUSED_HOUR_APPR_PROCESS_START_MEAN"])) * 1.)) > ((((data["REFUSED_DAYS_DECISION_MEAN"]) < ((-1.0 * ((data["REFUSED_DAYS_DECISION_MAX"]))))) * 1.))) * 1.))
    v["i220"] = 0.096999 * np.tanh((-1.0 * ((np.where(((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) - ((((data["NEW_SCORES_STD"]) < (data["ACTIVE_AMT_ANNUITY_MAX"])) * 1.))) > 0, data[
                                   "NEW_SCORES_STD"], ((((-1.0 * ((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"])))) < ((((data["ORGANIZATION_TYPE_Advertising"]) < (data["ACTIVE_AMT_ANNUITY_MAX"])) * 1.))) * 1.))))))
    v["i221"] = 0.098925 * np.tanh(((((((((((np.maximum(((data["BURO_MONTHS_BALANCE_MIN_MIN"])), ((data["BURO_STATUS_1_MEAN_MEAN"])))) > (data["BURO_STATUS_C_MEAN_MEAN"])) * 1.))
                                         < ((((data["BURO_STATUS_1_MEAN_MEAN"]) + (((0.318310) * (((data["BURO_MONTHS_BALANCE_MIN_MIN"]) * 2.0))))) / 2.0))) * 1.)) * 2.0)) * 2.0))
    v["i222"] = 0.099960 * np.tanh(((np.minimum(((data["ORGANIZATION_TYPE_Advertising"])), ((((((data["NAME_FAMILY_STATUS_Civil_marriage"]) + (np.where(data["AMT_ANNUITY"] < 0, data[
                                   "AMT_ANNUITY"], ((data["NAME_FAMILY_STATUS_Civil_marriage"]) + (((data["AMT_ANNUITY"]) * 2.0))))))) * (data["ORGANIZATION_TYPE_Advertising"])))))) * 2.0))
    v["i223"] = 0.096983 * np.tanh(np.minimum(((((np.tanh(((((data["HOUR_APPR_PROCESS_START"]) > (((np.tanh((data["AMT_ANNUITY"]))) / 2.0))) * 1.)))) / 2.0))),
                                              (((((((((((3.0)) / 2.0)) > (data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.)) > (data["HOUR_APPR_PROCESS_START"])) * 1.)))))
    v["i224"] = 0.070000 * np.tanh(((((((((((data["NEW_CREDIT_TO_INCOME_RATIO"]) * (data["DAYS_BIRTH"]))) > (0.318310)) * 1.)) + (np.minimum(((np.minimum(
        ((data["NEW_CREDIT_TO_INCOME_RATIO"])), ((data["AMT_ANNUITY"]))))), (((((data["DAYS_BIRTH"]) > (data["NEW_CREDIT_TO_INCOME_RATIO"])) * 1.)))))) / 2.0)) / 2.0))
    v["i225"] = 0.091985 * np.tanh(((np.where(data["OCCUPATION_TYPE_Secretaries"] > 0, data["AMT_ANNUITY"], (((3.141593) < (data["AMT_ANNUITY"])) * 1.))) + (np.where(
        data["AMT_ANNUITY"] < 0, data["OCCUPATION_TYPE_Secretaries"], np.where(data["OCCUPATION_TYPE_Secretaries"] > 0, -1.0, data["ORGANIZATION_TYPE_Advertising"])))))
    v["i226"] = 0.095139 * np.tanh((-1.0 * (((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (((((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, data["NAME_EDUCATION_TYPE_Academic_degree"],
                                                                                                     (-1.0 * ((((data["NAME_EDUCATION_TYPE_Academic_degree"]) * 2.0)))))) + ((((0.318310) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)))) * 2.0))) * 1.)))))
    v["i227"] = 0.098604 * np.tanh((((np.minimum(((((((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] > 0, data["ORGANIZATION_TYPE_Industry__type_11"], (4.0))) * (((data["ORGANIZATION_TYPE_Industry__type_11"]) - (
        data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) > (data["ORGANIZATION_TYPE_Industry__type_11"])) * 1.))
    v["i228"] = 0.100000 * np.tanh((((((((data["AMT_ANNUITY"]) + (((np.where(data["DAYS_BIRTH"] < 0, 0.0, (((2.0)) + (
        data["AMT_ANNUITY"])))) + (data["NEW_ANNUITY_TO_INCOME_RATIO"]))))) > ((6.81929636001586914))) * 1.)) * (data["AMT_ANNUITY"])))
    v["i229"] = 0.099554 * np.tanh((((((np.where(data["AMT_ANNUITY"] < 0, data["NAME_HOUSING_TYPE_With_parents"], ((((-1.0 * ((np.where(
        data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, (6.0), 1.570796))))) + (1.570796)) / 2.0))) - (-2.0))) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.))
    v["i230"] = 0.098967 * np.tanh(((data["DAYS_BIRTH"]) * ((-1.0 * ((((0.318310) * (((data["NAME_HOUSING_TYPE_With_parents"]) + (
        (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (np.minimum((((-1.0 * ((data["AMT_ANNUITY"]))))), ((0.0))))) * 1.)))))))))))
    v["i231"] = 0.061296 * np.tanh(((((np.maximum(((data["ORGANIZATION_TYPE_Industry__type_11"])), (((((data["ORGANIZATION_TYPE_Trade__type_4"]) + (data["AMT_ANNUITY"])) / 2.0))))) * (
        np.minimum((((((((data["ORGANIZATION_TYPE_Trade__type_4"]) + (data["AMT_ANNUITY"])) / 2.0)) / 2.0))), ((data["NAME_HOUSING_TYPE_With_parents"])))))) * (data["AMT_ANNUITY"])))
    v["i232"] = 0.098300 * np.tanh(np.where(data["NAME_HOUSING_TYPE_With_parents"] < 0, ((data["NAME_HOUSING_TYPE_With_parents"]) / 2.0), ((
        ((((data["AMT_ANNUITY"]) + ((((data["AMT_ANNUITY"]) < (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0))) * 1.))) / 2.0)) > ((((0.318310) < (data["AMT_ANNUITY"])) * 1.))) * 1.)))
    v["i233"] = 0.098259 * np.tanh(((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, (((data["WEEKDAY_APPR_PROCESS_START_FRIDAY"]) + (
        ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["ORGANIZATION_TYPE_Transport__type_3"])))) / 2.0), data["ORGANIZATION_TYPE_Transport__type_3"])))) - (data["ORGANIZATION_TYPE_Trade__type_5"])))
    v["i234"] = 0.099996 * np.tanh((((((-1.0 * ((np.where((((data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"]) > (1.570796)) * 1.) > 0, ((((data["AMT_ANNUITY"]) * 2.0)) * 2.0),
                                                          np.where(data["AMT_ANNUITY"] < 0, data["ORGANIZATION_TYPE_Transport__type_1"], data["ORGANIZATION_TYPE_Trade__type_4"])))))) * 2.0)) * 2.0))
    v["i235"] = 0.099031 * np.tanh(np.where(np.where(data["OCCUPATION_TYPE_Security_staff"] < 0, data["NEW_CAR_TO_BIRTH_RATIO"], data["NAME_EDUCATION_TYPE_Academic_degree"]) < -99998, data["OCCUPATION_TYPE_Security_staff"],
                                            (((data["AMT_ANNUITY"]) + (((data["NEW_CAR_TO_BIRTH_RATIO"]) + (np.where(data["AMT_ANNUITY"] < 0, data["NEW_CAR_TO_BIRTH_RATIO"], data["OCCUPATION_TYPE_Security_staff"]))))) / 2.0)))
    v["i236"] = 0.098701 * np.tanh(((((((((np.where(data["AMT_ANNUITY"] < -99998, data["AMT_ANNUITY"], ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])
                                                                                                        * ((((((((0.636620) * 2.0)) * 2.0)) < (data["AMT_ANNUITY"])) * 1.))))) * 2.0)) * 2.0)) * 2.0)) * 2.0))
    v["i237"] = 0.085625 * np.tanh(((((((np.minimum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), (((((-1.0 * ((data["ORGANIZATION_TYPE_Trade__type_4"])))) * 2.0))))) / 2.0)) + (
        (((((np.tanh((data["OCCUPATION_TYPE_Waiters_barmen_staff"]))) + (-1.0))) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)))) * 2.0))
    v["i238"] = 0.098000 * np.tanh(np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] > 0, (((-2.0) > (data["NEW_DOC_IND_KURT"])) * 1.), np.where(data["NEW_DOC_IND_KURT"]
                                                                                                                                            > 0, (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (data["REGION_POPULATION_RELATIVE"])) * 1.), ((((-2.0) * (data["REGION_POPULATION_RELATIVE"]))) / 2.0))))
    v["i239"] = 0.094999 * np.tanh(np.where(data["WALLSMATERIAL_MODE_Wooden"] < 0, np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, np.where(data["WEEKDAY_APPR_PROCESS_START_MONDAY"] < 0, data["NAME_INCOME_TYPE_Student"], data[
                                   "REGION_POPULATION_RELATIVE"]), (((data["REGION_POPULATION_RELATIVE"]) > (3.0)) * 1.)), np.minimum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), ((data["REGION_POPULATION_RELATIVE"])))))
    v["i240"] = 0.078810 * np.tanh(np.where(data["NEW_DOC_IND_KURT"] < 0, np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, ((
        data["OCCUPATION_TYPE_Medicine_staff"]) - (data["AMT_ANNUITY"])), data["AMT_CREDIT"]), (-1.0 * (((((data["AMT_CREDIT"]) > ((2.0))) * 1.))))))
    v["i241"] = 0.099910 * np.tanh(((((((((((np.tanh((1.570796))) < (data["CC_COUNT"])) * 1.)) > (((((data["CC_COUNT"]) - (
        data["CC_SK_DPD_DEF_SUM"]))) * ((((data["NAME_INCOME_TYPE_Student"]) < (data["CC_SK_DPD_DEF_SUM"])) * 1.))))) * 1.)) * 2.0)) * 2.0))
    v["i242"] = 0.094993 * np.tanh(((data["CC_AMT_PAYMENT_CURRENT_VAR"]) - (np.where((((np.minimum(((data["CC_AMT_PAYMENT_CURRENT_MEAN"])), ((data["CC_AMT_DRAWINGS_OTHER_CURRENT_VAR"])))) > (
        data["CC_AMT_DRAWINGS_ATM_CURRENT_VAR"])) * 1.) > 0, 2.0, ((((data["CC_AMT_PAYMENT_CURRENT_MEAN"]) * 2.0)) - (data["CC_CNT_DRAWINGS_POS_CURRENT_VAR"]))))))
    v["i243"] = 0.098760 * np.tanh(np.minimum(((np.maximum(((np.where(data["APPROVED_AMT_DOWN_PAYMENT_MAX"] < -99998, ((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]) - (data["CC_AMT_PAYMENT_CURRENT_MEAN"])), data[
                                   "CC_AMT_PAYMENT_CURRENT_MEAN"]))), ((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]))))), ((((data["CC_AMT_PAYMENT_CURRENT_MEAN"]) * (((data["CC_AMT_PAYMENT_CURRENT_VAR"]) - (2.0))))))))
    v["i244"] = 0.068305 * np.tanh(np.where(data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"] < -99998, data["NAME_CONTRACT_TYPE_Cash_loans"], ((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) - (
        (4.46922969818115234)))) * ((((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) + (data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"]))) > (((1.570796) - (data["NAME_CONTRACT_TYPE_Cash_loans"])))) * 1.)))))
    v["i245"] = 0.098999 * np.tanh(((data["EXT_SOURCE_3"]) * ((-1.0 * ((np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"] > 0, ((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) + (
        (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])) * 1.))), (((data["EXT_SOURCE_3"]) > (1.570796)) * 1.))))))))
    v["i246"] = 0.076350 * np.tanh(((((data["EXT_SOURCE_3"]) * 2.0)) * ((((np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] < 0, ((((-1.0 * ((data["EXT_SOURCE_3"]))))
                                                                                                                               > (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.), (-1.0 * ((data["AMT_ANNUITY"]))))) < (data["ACTIVE_DAYS_CREDIT_VAR"])) * 1.))))
    v["i247"] = 0.099104 * np.tanh((((((np.tanh((data["EXT_SOURCE_3"]))) > ((((data["NAME_EDUCATION_TYPE_Academic_degree"]) + (data["NEW_EXT_SOURCES_MEAN"])) / 2.0))) * 1.)) - (
        (((((data["NAME_EDUCATION_TYPE_Academic_degree"]) < ((-1.0 * ((data["EXT_SOURCE_3"]))))) * 1.)) + (data["NEW_EXT_SOURCES_MEAN"])))))
    v["i248"] = 0.090002 * np.tanh(((((data["DAYS_BIRTH"]) + ((((-1.0) + (data["BURO_CREDIT_TYPE_Car_loan_MEAN"])) / 2.0)))) * (np.minimum(
        ((((data["BURO_CREDIT_TYPE_Consumer_credit_MEAN"]) - (data["BURO_CREDIT_TYPE_Car_loan_MEAN"])))), (((((((data["NEW_SCORES_STD"]) * 2.0)) > (data["DAYS_BIRTH"])) * 1.)))))))
    v["i249"] = 0.099966 * np.tanh((-1.0 * ((((data["NEW_EXT_SOURCES_MEAN"]) - (
        (((((((np.tanh((((np.tanh((((((data["DAYS_BIRTH"]) * 2.0)) * 2.0)))) * 2.0)))) * 2.0)) * (data["CODE_GENDER"]))) + (-1.0)) / 2.0)))))))
    v["i250"] = 0.098702 * np.tanh(((((((data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]) > (data["FLAG_EMP_PHONE"])) * 1.)) > (
        (((0.636620) + (((0.636620) + (np.minimum(((data["REGION_RATING_CLIENT"])), ((data["DAYS_BIRTH"]))))))) / 2.0))) * 1.))
    v["i251"] = 0.099990 * np.tanh((-1.0 * ((np.where((-1.0 * ((data["NEW_SCORES_STD"]))) > 0, data["REGION_RATING_CLIENT"], ((data["NEW_EXT_SOURCES_MEAN"]) + (
        (((data["NEW_SCORES_STD"]) > ((((((data["NEW_EXT_SOURCES_MEAN"]) + (0.636620))) > (data["REGION_RATING_CLIENT"])) * 1.))) * 1.))))))))
    v["i252"] = 0.099230 * np.tanh(((np.where(data["NEW_CREDIT_TO_INCOME_RATIO"] > 0, np.minimum((((((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) < (((data["REG_CITY_NOT_WORK_CITY"]) / 2.0))) * 1.))),
                                                                                                 ((((data["NEW_CREDIT_TO_INCOME_RATIO"]) + (data["REGION_RATING_CLIENT"]))))), (-1.0 * ((((data["REG_CITY_NOT_WORK_CITY"]) / 2.0)))))) / 2.0))
    v["i253"] = 0.099500 * np.tanh(((((data["NEW_EXT_SOURCES_MEAN"]) + ((13.18767356872558594)))) * ((((np.where(data["NEW_EXT_SOURCES_MEAN"] > 0, ((
        ((((data["NEW_EXT_SOURCES_MEAN"]) + (-1.0)) / 2.0)) < (data["CLOSED_MONTHS_BALANCE_SIZE_SUM"])) * 1.), -2.0)) > (data["NEW_EXT_SOURCES_MEAN"])) * 1.))))
    v["i254"] = 0.099742 * np.tanh(((((3.0) - (np.where(np.where(data["EXT_SOURCE_1"] < -99998, ((3.0) - (
        data["ACTIVE_AMT_ANNUITY_MEAN"])), data["DAYS_BIRTH"]) < 0, data["EXT_SOURCE_1"], 3.141593)))) * 2.0))
    v["i255"] = 0.099548 * np.tanh(np.where(data["NEW_SOURCES_PROD"] < -99998, np.where(data["NEW_EXT_SOURCES_MEAN"] < -99998, data["NEW_SOURCES_PROD"], ((
        np.tanh((np.tanh((data["NEW_EXT_SOURCES_MEAN"]))))) - (data["NEW_EXT_SOURCES_MEAN"]))), np.minimum(((0.636620)), ((data["NEW_EXT_SOURCES_MEAN"])))))
    v["i256"] = 0.089995 * np.tanh(np.where(data["AMT_REQ_CREDIT_BUREAU_QRT"] > 0, np.minimum(((((data["AMT_REQ_CREDIT_BUREAU_QRT"]) * (data["BURO_DAYS_CREDIT_ENDDATE_MEAN"])))), ((
        data["AMT_REQ_CREDIT_BUREAU_QRT"]))), (((data["EXT_SOURCE_3"]) > (((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]) - (np.minimum(((data["AMT_REQ_CREDIT_BUREAU_YEAR"])), ((-1.0))))))) * 1.)))
    v["i257"] = 0.099999 * np.tanh(((np.tanh((np.minimum(((((np.where(data["EXT_SOURCE_3"] < -99998, data["HOUR_APPR_PROCESS_START"], np.maximum(((data["NEW_INC_BY_ORG"])), ((
        data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]))))) + (data["HOUR_APPR_PROCESS_START"])))), ((((data["EXT_SOURCE_3"]) * (data["NEW_INC_BY_ORG"])))))))) / 2.0))
    v["i258"] = 0.099998 * np.tanh(np.minimum(((0.318310)), ((((((((data["CLOSED_MONTHS_BALANCE_MAX_MAX"]) > ((((data["CLOSED_MONTHS_BALANCE_MAX_MAX"]) < ((((data["ACTIVE_AMT_ANNUITY_MAX"]) < (
        (-1.0 * ((data["CLOSED_MONTHS_BALANCE_MAX_MAX"]))))) * 1.))) * 1.))) * 1.)) < ((((data["NEW_INC_BY_ORG"]) > (data["OCCUPATION_TYPE_Cooking_staff"])) * 1.))) * 1.)))))
    v["i259"] = 0.044980 * np.tanh(np.maximum(((np.where(data["LIVINGAREA_MODE"] < 0, data["OCCUPATION_TYPE_Medicine_staff"], (((np.where(data["DAYS_ID_PUBLISH"] < 0, 1.570796, data[
                                   "OCCUPATION_TYPE_High_skill_tech_staff"])) > (data["OCCUPATION_TYPE_Medicine_staff"])) * 1.)))), ((((-2.0) - (data["NEW_EXT_SOURCES_MEAN"]))))))
    v["i260"] = 0.099700 * np.tanh(np.where(data["BURO_DAYS_CREDIT_ENDDATE_MEAN"] < -99998, ((data["AMT_INCOME_TOTAL"]) * 2.0), np.where((((data["DAYS_ID_PUBLISH"]) + (
        data["BURO_DAYS_CREDIT_ENDDATE_MEAN"])) / 2.0) < 0, ((0.636620) / 2.0), np.minimum(((data["OCCUPATION_TYPE_High_skill_tech_staff"])), (((-1.0 * ((data["AMT_INCOME_TOTAL"])))))))))
    v["i261"] = 0.099990 * np.tanh((((((((data["NEW_EXT_SOURCES_MEAN"]) < (((np.tanh((data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"]))) - (
        ((((((data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"]) > (((data["CLOSED_MONTHS_BALANCE_MIN_MIN"]) - (data["CLOSED_MONTHS_BALANCE_MAX_MAX"])))) * 1.)) + (3.141593)) / 2.0))))) * 1.)) * 2.0)) * 2.0))
    v["i262"] = 0.094001 * np.tanh(np.minimum((((((data["NEW_EXT_SOURCES_MEAN"]) > (0.636620)) * 1.))), (((((11.55127620697021484)) * (
        (((((((0.636620) > (((data["EXT_SOURCE_1"]) / 2.0))) * 1.)) / 2.0)) - (((data["NEW_EXT_SOURCES_MEAN"]) / 2.0)))))))))
    v["i263"] = 0.099700 * np.tanh((-1.0 * (((((data["BURO_CREDIT_TYPE_Car_loan_MEAN"]) > (np.where(np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, data[
                                   "BURO_CREDIT_TYPE_Loan_for_business_development_MEAN"], 3.141593) > 0, (-1.0 * ((data["BURO_CREDIT_ACTIVE_Closed_MEAN"]))), ((3.141593) + (0.636620))))) * 1.)))))
    v["i264"] = 0.096890 * np.tanh(((np.minimum(((np.minimum(((((data["NAME_EDUCATION_TYPE_Higher_education"]) / 2.0))), ((data["NEW_EXT_SOURCES_MEAN"]))))), ((data["AMT_INCOME_TOTAL"])))) * (
        (-1.0 * (((((np.where(data["NEW_CAR_TO_EMPLOY_RATIO"] < 0, data["AMT_INCOME_TOTAL"], 3.141593)) + (data["NAME_EDUCATION_TYPE_Higher_education"])) / 2.0)))))))
    v["i265"] = 0.099603 * np.tanh(((((np.where(data["NEW_CAR_TO_EMPLOY_RATIO"] < -99998, np.tanh((((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]) * 2.0))), ((
        data["CODE_GENDER"]) + (((((data["NEW_CAR_TO_EMPLOY_RATIO"]) + (data["NEW_CAR_TO_EMPLOY_RATIO"]))) + (data["NEW_CAR_TO_EMPLOY_RATIO"])))))) / 2.0)) / 2.0))
    v["i266"] = 0.083000 * np.tanh(np.maximum((((((((data["CLOSED_MONTHS_BALANCE_MIN_MIN"]) > (data["NEW_EXT_SOURCES_MEAN"])) * 1.)) * (data["NEW_EXT_SOURCES_MEAN"])))), ((np.where(
        data["NEW_EXT_SOURCES_MEAN"] > 0, np.where(data["CLOSED_MONTHS_BALANCE_MIN_MIN"] > 0, data["EXT_SOURCE_1"], data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]), ((data["EXT_SOURCE_1"]) * 2.0))))))
    v["i267"] = 0.097003 * np.tanh(np.minimum(((((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) * ((((data["OCCUPATION_TYPE_Cooking_staff"]) + (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])) / 2.0))))), ((
        (((((((-1.0 * ((data["NAME_TYPE_SUITE_Other_A"])))) * (data["OCCUPATION_TYPE_Cooking_staff"]))) - (data["NAME_TYPE_SUITE_Other_A"]))) - (data["ORGANIZATION_TYPE_Trade__type_4"]))))))
    v["i268"] = 0.097000 * np.tanh(((((((np.where(data["NEW_EXT_SOURCES_MEAN"] < -99998, data["WEEKDAY_APPR_PROCESS_START_TUESDAY"], ((
        (0.636620) > (((3.0) + (((data["NEW_EXT_SOURCES_MEAN"]) + (data["NAME_INCOME_TYPE_Maternity_leave"])))))) * 1.))) * 2.0)) * 2.0)) * 2.0))
    v["i269"] = 0.076496 * np.tanh(((((data["FLAG_DOCUMENT_3"]) * (np.minimum(((0.0)), ((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]
                                                                                                  > 0, (((((-1.0) - (data["AMT_ANNUITY"]))) + (np.tanh((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) / 2.0), 0.636620))))))) * 2.0))
    v["i270"] = 0.099960 * np.tanh(np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] > 0, (((data["NEW_EXT_SOURCES_MEAN"]) < (-1.0)) * 1.), ((np.where(data["NEW_EXT_SOURCES_MEAN"] < 0, data[
                                   "ORGANIZATION_TYPE_Trade__type_3"], (-1.0 * ((data["ORGANIZATION_TYPE_Trade__type_3"]))))) - ((((1.570796) < (data["NEW_EXT_SOURCES_MEAN"])) * 1.)))))
    v["i271"] = 0.096900 * np.tanh(np.where(data["BURO_AMT_ANNUITY_MAX"] < 0, np.where(data["ORGANIZATION_TYPE_Trade__type_3"] < 0, (((0.636620) < (((data["ACTIVE_MONTHS_BALANCE_MIN_MIN"]) / 2.0)))
                                                                                                                                     * 1.), ((((data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"]) * 2.0)) * 2.0)), (((data["BURO_AMT_ANNUITY_MAX"]) > (data["NEW_EXT_SOURCES_MEAN"])) * 1.)))
    v["i272"] = 0.056000 * np.tanh((((((((((-1.0 * ((np.maximum(((np.where(data["EXT_SOURCE_1"] < 0, (-1.0 * ((np.maximum(((data["APPROVED_AMT_DOWN_PAYMENT_MAX"])), ((
        data["NAME_EDUCATION_TYPE_Academic_degree"])))))), data["APPROVED_AMT_DOWN_PAYMENT_MEAN"]))), ((data["NAME_EDUCATION_TYPE_Academic_degree"]))))))) * 2.0)) * 2.0)) * 2.0)) * 2.0))
    v["i273"] = 0.095900 * np.tanh(np.maximum((((((((1.570796) < (((data["NEW_CAR_TO_BIRTH_RATIO"]) * 2.0))) * 1.)) * ((-1.0 * ((data["NEW_EXT_SOURCES_MEAN"]))))))),
                                              ((np.where(((data["NEW_EXT_SOURCES_MEAN"]) - (1.570796)) > 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], -2.0)))))
    v["i274"] = 0.065300 * np.tanh((((((((((((data["NEW_CAR_TO_BIRTH_RATIO"]) < ((((data["NEW_CAR_TO_BIRTH_RATIO"]) > (data["OWN_CAR_AGE"])) * 1.))) * 1.))
                                          > (data["NEW_EXT_SOURCES_MEAN"])) * 1.)) < (((data["OWN_CAR_AGE"]) * (((1.570796) / 2.0))))) * 1.)) * (1.570796)))
    v["i275"] = 0.098970 * np.tanh(((data["NEW_CAR_TO_EMPLOY_RATIO"]) * ((((0.636620) < (((data["NEW_CAR_TO_BIRTH_RATIO"]) - (
        ((data["NEW_CAR_TO_EMPLOY_RATIO"]) * ((((data["NEW_CAR_TO_BIRTH_RATIO"]) < (((data["NEW_EXT_SOURCES_MEAN"]) * (data["NEW_EXT_SOURCES_MEAN"])))) * 1.))))))) * 1.))))
    v["i276"] = 0.050995 * np.tanh((((((((((((data["WEEKDAY_APPR_PROCESS_START_WEDNESDAY"]) > (((data["AMT_ANNUITY"]) * (data["WEEKDAY_APPR_PROCESS_START_WEDNESDAY"])))) * 1.)) - (
        data["NAME_INCOME_TYPE_Student"]))) - (np.where(data["WEEKDAY_APPR_PROCESS_START_WEDNESDAY"] < 0, data["NAME_INCOME_TYPE_Student"], data["AMT_ANNUITY"])))) / 2.0)) / 2.0))
    v["i277"] = 0.099600 * np.tanh(np.where(data["NEW_EXT_SOURCES_MEAN"] < -99998, data["NEW_EXT_SOURCES_MEAN"], ((((((-1.0) < (np.where(data["OWN_CAR_AGE"] > 0, data[
                                   "NEW_ANNUITY_TO_INCOME_RATIO"], np.maximum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((data["NEW_CAR_TO_BIRTH_RATIO"])))))) * 1.)) < (((data["NEW_EXT_SOURCES_MEAN"]) / 2.0))) * 1.)))
    v["i278"] = 0.098097 * np.tanh(((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (((3.0) * (np.where((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (3.0)) * 1.) > 0, data[
                                   "NEW_EXT_SOURCES_MEAN"], (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (((data["NEW_EXT_SOURCES_MEAN"]) + (3.141593)))) * 1.)))))))
    v["i279"] = 0.099890 * np.tanh(np.where(data["NEW_CAR_TO_EMPLOY_RATIO"] < -99998, data["NAME_INCOME_TYPE_Maternity_leave"], np.minimum(((((((0.318310) * 2.0)) + (
        data["NEW_ANNUITY_TO_INCOME_RATIO"])))), (((((-1.0 * (((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (data["NEW_CAR_TO_EMPLOY_RATIO"])) * 1.))))) / 2.0))))))
    v["i280"] = 0.093509 * np.tanh(np.where(((np.tanh((np.tanh((data["OWN_CAR_AGE"]))))) - (data["CNT_CHILDREN"])) < 0, ((
        (data["FLAG_OWN_CAR"]) > (data["CNT_CHILDREN"])) * 1.), np.tanh((np.tanh((np.tanh((np.tanh((data["OWN_CAR_AGE"]))))))))))
    v["i281"] = 0.091999 * np.tanh((((((((data["AMT_ANNUITY"]) > ((2.73518514633178711))) * 1.)) * 2.0)) + (((np.where(data["NEW_EXT_SOURCES_MEAN"] < -99998, data[
                                   "NEW_EXT_SOURCES_MEAN"], (((data["NEW_EXT_SOURCES_MEAN"]) < ((-1.0 * (((2.73518514633178711)))))) * 1.))) + (data["ORGANIZATION_TYPE_Telecom"])))))
    v["i282"] = 0.080945 * np.tanh(np.where(np.where(((((data["AMT_ANNUITY"]) * (np.minimum(((data["AMT_ANNUITY"])), (((-1.0 * ((data["NAME_EDUCATION_TYPE_Academic_degree"]))))))))) + (
        data["NAME_INCOME_TYPE_Maternity_leave"])) > 0, data["ORGANIZATION_TYPE_Trade__type_4"], 3.0) < 0, data["ORGANIZATION_TYPE_Trade__type_4"], data["WEEKDAY_APPR_PROCESS_START_TUESDAY"]))
    v["i283"] = 0.098110 * np.tanh(((((((data["AMT_ANNUITY"]) + ((((data["NAME_EDUCATION_TYPE_Academic_degree"]) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)))) + (np.minimum(
        ((data["NAME_EDUCATION_TYPE_Academic_degree"])), ((data["EMERGENCYSTATE_MODE_Yes"])))))) * (((data["ORGANIZATION_TYPE_Transport__type_2"]) - (data["EMERGENCYSTATE_MODE_Yes"])))))
    v["i284"] = 0.059900 * np.tanh((((np.maximum(((((((1.570796) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))), ((
        ((np.tanh((((data["ORGANIZATION_TYPE_Advertising"]) / 2.0)))) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))) > (1.570796)) * 1.))
    v["i285"] = 0.088980 * np.tanh(np.minimum(((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["INSTAL_PAYMENT_PERC_MEAN"])))), (((-1.0 * ((np.where(
        data["INSTAL_PAYMENT_PERC_MEAN"] > 0, 3.141593, (((data["INSTAL_PAYMENT_PERC_MEAN"]) + (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["INSTAL_PAYMENT_PERC_MEAN"])))) / 2.0)))))))))
    v["i286"] = 0.058000 * np.tanh(np.minimum(((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, ((data["ORGANIZATION_TYPE_Legal_Services"]) / 2.0), data[
                                   "OCCUPATION_TYPE_High_skill_tech_staff"]))), (((-1.0 * ((((((((-2.0) * 2.0)) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)))) * 2.0))))))))
    v["i287"] = 0.099990 * np.tanh((((((((-1.0 * ((((((((2.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) / 2.0)) < (((0.318310) + (data["NAME_INCOME_TYPE_Maternity_leave"])))) * 1.))))) - (
        data["NAME_EDUCATION_TYPE_Academic_degree"]))) - (data["NAME_EDUCATION_TYPE_Academic_degree"]))) - (data["NAME_EDUCATION_TYPE_Academic_degree"])))
    v["i288"] = 0.089938 * np.tanh(((np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])), ((np.minimum(((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, (((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (
        data["NAME_INCOME_TYPE_Maternity_leave"])) / 2.0)) / 2.0), data["AMT_ANNUITY"]))), ((data["NAME_INCOME_TYPE_Maternity_leave"]))))))) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))
    v["i289"] = 0.099996 * np.tanh(((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] > 0, ((data["AMT_ANNUITY"]) - ((((data["AMT_ANNUITY"]) < (
        ((2.0) / 2.0))) * 1.))), (((data["AMT_ANNUITY"]) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.))) * (((data["DAYS_REGISTRATION"]) / 2.0))))
    v["i290"] = 0.099996 * np.tanh(((((((((data["WALLSMATERIAL_MODE_Stone__brick"]) + ((-1.0 * ((np.minimum((((-1.0 * ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))),
                                                                                                            ((np.minimum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))))))))) / 2.0)) * 2.0)) > (3.0)) * 1.))
    v["i291"] = 0.063400 * np.tanh((((-1.0 * (((((((np.minimum(((data["NONLIVINGAREA_MEDI"])), (((-1.0 * ((data["NONLIVINGAREA_AVG"]))))))) > (
        ((data["NAME_INCOME_TYPE_Maternity_leave"]) - (0.318310)))) * 1.)) * (((data["WALLSMATERIAL_MODE_Stone__brick"]) / 2.0))))))) * 2.0))
    v["i292"] = 0.076349 * np.tanh(((data["AMT_ANNUITY"]) * ((-1.0 * (((((data["YEARS_BUILD_AVG"]) > ((((data["YEARS_BUILD_AVG"]) + (((((((data["AMT_ANNUITY"]) > (
        data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.)) > (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) * 1.))) / 2.0))) * 1.)))))))
    v["i293"] = 0.099173 * np.tanh((((((data["LIVINGAPARTMENTS_AVG"]) > (np.where((-1.0 * ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) > 0, (5.0),
                                                                                  np.maximum((((-1.0 * (((((3.0) > (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0))) * 1.)))))), ((data["AMT_ANNUITY"])))))) * 1.)) * 2.0))
    v["i294"] = 0.099970 * np.tanh((-1.0 * (((((np.where(data["LIVINGAPARTMENTS_AVG"] < -99998, data["YEARS_BUILD_MODE"], data["COMMONAREA_MEDI"])) > (
        np.minimum((((((data["YEARS_BUILD_MODE"]) < ((1.0))) * 1.))), ((np.tanh((((data["YEARS_BUILD_MODE"]) * 2.0)))))))) * 1.)))))
    v["i295"] = 0.099718 * np.tanh((((((data["FLOORSMIN_AVG"]) > (np.where(data["LIVINGAREA_AVG"] > 0, np.maximum(((0.636620)), ((data["LIVINGAREA_MODE"]))),
                                                                           ((data["LIVINGAREA_AVG"]) * (data["FLOORSMIN_AVG"]))))) * 1.)) * ((((10.97417545318603516)) * (data["LIVINGAREA_AVG"])))))
    v["i296"] = 0.079954 * np.tanh(np.where(data["WALLSMATERIAL_MODE_Stone__brick"] < 0, data["NAME_INCOME_TYPE_Maternity_leave"], ((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (
        ((np.where((((((data["LIVINGAREA_MEDI"]) * 2.0)) > (data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.) > 0, (3.0), data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 2.0)))))
    v["i297"] = 0.098995 * np.tanh(((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (np.minimum(((np.where(((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["AMT_ANNUITY"])) > 0, data[
                                   "ORGANIZATION_TYPE_Advertising"], data["AMT_ANNUITY"]))), ((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["AMT_ANNUITY"]))) - (data["EXT_SOURCE_1"]))))))))
    v["i298"] = 0.079895 * np.tanh(np.where(data["NEW_CAR_TO_EMPLOY_RATIO"] < 0, ((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (
        2.0)) * 1.)))) - (data["WALLSMATERIAL_MODE_Monolithic"]))) - (data["NAME_INCOME_TYPE_Student"])), data["WALLSMATERIAL_MODE_Monolithic"]))
    v["i299"] = 0.099500 * np.tanh((((((-2.0) > (np.where(data["EXT_SOURCE_3"] < -99998, ((-2.0) * (data["AMT_ANNUITY"])), ((
        np.maximum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), (((((-1.0 * ((data["EXT_SOURCE_3"])))) * 2.0))))) * 2.0)))) * 1.)) * 2.0))
    v["i300"] = 0.099751 * np.tanh(np.where(data["CLOSED_DAYS_CREDIT_VAR"] > 0, np.minimum(((((((((((data["AMT_ANNUITY"]) * (
        data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]))) * 2.0)) * 2.0)) * 2.0))), ((((data["BURO_DAYS_CREDIT_VAR"]) * (data["AMT_ANNUITY"]))))), (0.0)))
    v["i301"] = 0.057002 * np.tanh(((((((((data["BURO_DAYS_CREDIT_VAR"]) / 2.0)) < (data["EXT_SOURCE_3"])) * 1.)) < (
        ((((-1.0 * ((0.318310)))) < (np.minimum(((((data["BURO_DAYS_CREDIT_VAR"]) * 2.0))), ((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]))))) * 1.))) * 1.))
    v["i302"] = 0.094999 * np.tanh((((((((-1.0 * ((np.maximum(((data["CLOSED_DAYS_CREDIT_VAR"])), ((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]))))))) < (
        ((1.0) - (np.minimum(((0.636620)), ((1.0))))))) * 1.)) < ((((data["EXT_SOURCE_3"]) > (data["BURO_DAYS_CREDIT_VAR"])) * 1.))) * 1.))
    v["i303"] = 0.097200 * np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_MEAN"] < 0, np.where(data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"] < 0, ((((1.09531068801879883)) < (((data["ACTIVE_DAYS_CREDIT_MEAN"]) + (
        (((data["ACTIVE_AMT_CREDIT_SUM_MEAN"]) < (data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"])) * 1.))))) * 1.), data["ACTIVE_DAYS_CREDIT_MEAN"]), ((data["ACTIVE_DAYS_CREDIT_MEAN"]) * 2.0)))
    v["i304"] = 0.099712 * np.tanh(np.where(data["CLOSED_AMT_CREDIT_SUM_SUM"] > 0, np.where(data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"] > 0, -2.0, ((((data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]) / 2.0)) * (
        data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]))), np.where(data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"] > 0, data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"], (-1.0 * ((0.318310))))))
    v["i305"] = 0.099669 * np.tanh(((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * ((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) > (((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) * (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])))) * 1.))))
                                    * ((((data["CLOSED_DAYS_CREDIT_VAR"]) > (np.minimum(((((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) * 2.0))), ((data["NEW_ANNUITY_TO_INCOME_RATIO"]))))) * 1.))))
    v["i306"] = 0.024020 * np.tanh(np.minimum((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)) < (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])) * 1.))), ((
        ((np.minimum(((-2.0)), ((((-2.0) * (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])))))) * (data["BURO_CREDIT_TYPE_Mortgage_MEAN"]))))))
    v["i307"] = 0.097699 * np.tanh((((((((((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) > (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])) * 1.)) / 2.0)) - (data["BURO_CREDIT_TYPE_Mortgage_MEAN"]))) * 2.0))
                                    * (np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])), ((((((-1.0 * ((data["BURO_CREDIT_TYPE_Mortgage_MEAN"])))) < (data["CLOSED_DAYS_CREDIT_VAR"])) * 1.)))))))
    v["i308"] = 0.095950 * np.tanh(((np.where(data["BURO_CREDIT_TYPE_Consumer_credit_MEAN"] > 0, np.where(data["BURO_DAYS_CREDIT_MEAN"] < 0, 0.0, data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]),
                                              (-1.0 * ((((((data["BURO_DAYS_CREDIT_MEAN"]) - (data["BURO_CREDIT_TYPE_Another_type_of_loan_MEAN"]))) * (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]))))))) / 2.0))
    v["i309"] = 0.068960 * np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"] > 0, data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"], ((np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"] < -
                                                                                                                                                   99998, (0.06772400438785553), (-1.0 * ((((((-1.0 * (((((0.06772400438785553)) - (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])))))) < (data["ACTIVE_AMT_CREDIT_SUM_MEAN"])) * 1.)))))) * 2.0)))
    v["i310"] = 0.097900 * np.tanh((((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) > (((np.tanh((np.tanh((data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]))))) / 2.0))) * 1.)) * (
        ((((((3.141593) - (data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]))) - (data["CLOSED_AMT_CREDIT_SUM_SUM"]))) - (data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"])))))
    v["i311"] = 0.058059 * np.tanh(np.where((((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) + (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])) / 2.0) > 0, (((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) + (
        data["AMT_ANNUITY"])) / 2.0), np.where(((0.318310) + (((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) * 2.0))) > 0, data["NEW_ANNUITY_TO_INCOME_RATIO"], data["NAME_INCOME_TYPE_Maternity_leave"])))
    v["i312"] = 0.004031 * np.tanh((((((((0.636620) < (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])) * 1.)) * (((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) - (
        (((0.636620) > (data["AMT_ANNUITY"])) * 1.)))) - ((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) > (data["AMT_CREDIT"])) * 1.)))))) * 2.0))
    v["i313"] = 0.029029 * np.tanh((((np.where((-1.0 * ((np.maximum(((data["AMT_ANNUITY"])), ((data["NEW_ANNUITY_TO_INCOME_RATIO"])))))) > 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"],
                                               np.minimum(((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])), ((data["AMT_CREDIT"]))))) > ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > ((-1.0 * ((1.0))))) * 1.))) * 1.))
    v["i314"] = 0.051999 * np.tanh(((np.minimum(((np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"] < 0, np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] < 0, ((1.0) - (data["AMT_CREDIT"])), data[
                                   "NEW_ANNUITY_TO_INCOME_RATIO"]), data["NEW_ANNUITY_TO_INCOME_RATIO"]))), (((((data["AMT_CREDIT"]) < (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])) * 1.))))) / 2.0))
    v["i315"] = 0.099950 * np.tanh(((np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"] > 0, (-1.0 * ((data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"]))), ((((((data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"]) + (
        np.where(data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"] > 0, data["NAME_INCOME_TYPE_Maternity_leave"], 2.0))) / 2.0)) < (((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]) * 2.0))) * 1.))) * 2.0))
    v["i316"] = 0.100000 * np.tanh(((np.where(data["ACTIVE_CREDIT_DAY_OVERDUE_MEAN"] > 0, (((-1.0 * ((((((data["ACTIVE_CREDIT_DAY_OVERDUE_MEAN"]) - (data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]))) - (
        data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"])))))) * ((8.35832691192626953))), (((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]) > ((5.0))) * 1.))) * 2.0))
    v["i317"] = 0.090700 * np.tanh(np.where(np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"] > 0, data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"], data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]) < 0, ((
        (data["ACTIVE_DAYS_CREDIT_MEAN"]) > (((((2.37051534652709961)) + (data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"])) / 2.0))) * 1.), ((data["ACTIVE_DAYS_CREDIT_MEAN"]) + (((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]) * 2.0)))))
    v["i318"] = 0.071000 * np.tanh(np.where(data["ACTIVE_CNT_CREDIT_PROLONG_SUM"] > 0, data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"], np.where((((data["ACTIVE_DAYS_CREDIT_MEAN"]) + (
        data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"])) / 2.0) < 0, 0.0, ((((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]) * 2.0)) - ((((data["ACTIVE_DAYS_CREDIT_MEAN"]) + (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])) / 2.0))))))
    v["i319"] = 0.096003 * np.tanh(np.where(data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"] > 0, data[
                                   "ORGANIZATION_TYPE_Trade__type_4"], ((np.tanh((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) * (0.318310))))
    v["i320"] = 0.099110 * np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"] < 0, np.where(data["ACTIVE_DAYS_CREDIT_MEAN"] < 0, data["NAME_INCOME_TYPE_Maternity_leave"], data[
                                   "ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]), ((((((((((np.tanh((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]))) * 2.0)) + (data["ACTIVE_DAYS_CREDIT_MEAN"]))) * 2.0)) * 2.0)) * 2.0)))
    v["i321"] = 0.099160 * np.tanh((((data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"]) > (((((((((data["NEW_CREDIT_TO_INCOME_RATIO"]) < (data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"])) * 1.)) - (
        ((data["ACTIVE_DAYS_CREDIT_MEAN"]) + (np.where(data["NEW_CREDIT_TO_INCOME_RATIO"] > 0, 0.318310, data["ACTIVE_DAYS_CREDIT_MEAN"])))))) < (3.0)) * 1.))) * 1.))
    v["i322"] = 0.083290 * np.tanh((((np.where(data["ACTIVE_DAYS_CREDIT_MEAN"] < 0, (((data["ACTIVE_DAYS_CREDIT_MEAN"]) > (np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"] < 0, data[
                                   "ACTIVE_DAYS_CREDIT_MEAN"], data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"]))) * 1.), data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"])) > (np.tanh((((1.0) / 2.0))))) * 1.))
    v["i323"] = 0.095882 * np.tanh(np.where(data["ACTIVE_CNT_CREDIT_PROLONG_SUM"] > 0, np.maximum(((data["ACTIVE_DAYS_CREDIT_MEAN"])), ((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]))), ((
        ((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]) * 2.0)) * (np.maximum(((np.where(data["ACTIVE_CREDIT_DAY_OVERDUE_MEAN"] < -99998, 0.0, data["ACTIVE_CREDIT_DAY_OVERDUE_MEAN"]))), ((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"])))))))
    v["i324"] = 0.099008 * np.tanh(((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]) * (np.maximum(((((((((-2.0) + ((((-2.0) + (data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"])) / 2.0))) / 2.0)) + (
        data["ACTIVE_CREDIT_DAY_OVERDUE_MEAN"])) / 2.0))), ((((data["ACTIVE_DAYS_CREDIT_MEAN"]) - (data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"]))))))))
    v["i325"] = 0.001009 * np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"] < 0, np.where(data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"] < -99998, ((0.318310) / 2.0), data[
                                   "ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]), ((((((((((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]) + (data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)))
    v["i326"] = 0.088000 * np.tanh(np.where(np.where(data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"] > 0, data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"], data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"])
                                            > 0, ((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]) + (data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"])), (-1.0 * ((((((2.65623164176940918)) < (((0.0) + (data["AMT_CREDIT"])))) * 1.))))))
    v["i327"] = 0.013070 * np.tanh(np.tanh((((data["ACTIVE_AMT_CREDIT_SUM_MEAN"]) * (((-2.0) * ((((np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"] < 0, data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"], data[
                                   "ACTIVE_DAYS_CREDIT_UPDATE_MEAN"])) > (np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"] > 0, 0.0, data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]))) * 1.))))))))
    v["i328"] = 0.099470 * np.tanh((((np.minimum(((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"])), ((data["ACTIVE_AMT_CREDIT_SUM_SUM"])))) < (((np.where(data["ACTIVE_AMT_CREDIT_SUM_SUM"] > 0, ((
        data["ACTIVE_CNT_CREDIT_PROLONG_SUM"]) - (data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"])), (((data["ACTIVE_CNT_CREDIT_PROLONG_SUM"]) + (data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"])) / 2.0))) * 2.0))) * 1.))
    v["i329"] = 0.091989 * np.tanh(((np.where(-2.0 < -99998, 0.318310, (((((0.318310) < (np.tanh((data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"]))))
                                                                          * 1.)) - ((((data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"]) > (np.tanh((-2.0)))) * 1.))))) / 2.0))
    v["i330"] = 0.096602 * np.tanh(((np.where(((np.tanh(((7.0)))) + (data["ACTIVE_DAYS_CREDIT_MEAN"])) > 0, ((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]) * (
        (7.0))), (((((2.0) + (data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"]))) < (data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"])) * 1.))) * 2.0))
    v["i331"] = 0.100000 * np.tanh(((np.maximum((((((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]) - (0.318310))) > (1.570796)) * 1.))), ((((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]) * 2.0))))) - (
        (((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]) > ((((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]) < (data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"])) * 1.))) * 1.))))
    v["i332"] = 0.091990 * np.tanh(np.where(data["ACTIVE_CREDIT_DAY_OVERDUE_MEAN"] > 0, -1.0, (((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]) < (np.minimum(((np.minimum(
        ((data["ACTIVE_CREDIT_DAY_OVERDUE_MEAN"])), ((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]))))), (((((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) + (((-1.0) / 2.0))) / 2.0)))))) * 1.)) * 2.0)))
    v["i333"] = 0.099995 * np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"] < 0, ((((((((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) > (data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"])) * 1.)) > (
        data["ACTIVE_AMT_CREDIT_SUM_SUM"])) * 1.)) * ((((((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) > (data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"])) * 1.)) * (data["ACTIVE_DAYS_CREDIT_MEAN"])))), data["ACTIVE_DAYS_CREDIT_MEAN"]))
    v["i334"] = 0.094326 * np.tanh(((((((((np.maximum(((2.0)), ((np.minimum(((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"])), ((data["ACTIVE_AMT_CREDIT_SUM_SUM"])))))))
                                          > (np.minimum(((data["ACTIVE_AMT_CREDIT_SUM_SUM"])), ((3.141593))))) * 1.)) + (-1.0)) / 2.0)) * ((6.0))))
    v["i335"] = 0.092296 * np.tanh((((np.where((((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) + (1.570796)) / 2.0) > 0, (((3.0) < (((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) + (
        np.maximum(((data["ACTIVE_MONTHS_BALANCE_SIZE_SUM"])), ((data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]))))))) * 1.), data["ACTIVE_MONTHS_BALANCE_SIZE_SUM"])) > (0.636620)) * 1.))
    v["i336"] = 0.016198 * np.tanh((-1.0 * ((((np.where(data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"] < 0, (((data["ACTIVE_DAYS_CREDIT_MEAN"]) > (0.318310)) * 1.), ((
        ((((data["ACTIVE_DAYS_CREDIT_MEAN"]) > (0.318310)) * 1.)) + ((((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]) + (data["ACTIVE_CNT_CREDIT_PROLONG_SUM"])) / 2.0))) / 2.0))) / 2.0)))))
    v["i337"] = 0.095499 * np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"] > 0, (((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) < (((((data["ACTIVE_CREDIT_DAY_OVERDUE_MEAN"]) * 2.0)) * 2.0)))
                                                                                         * 1.), np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_SUM"])), ((np.where(data["ACTIVE_DAYS_CREDIT_MEAN"] < 0, 0.0, data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]))))))
    v["i338"] = 0.099099 * np.tanh(np.where(data["ACTIVE_AMT_ANNUITY_MAX"] > 0, data["ACTIVE_CNT_CREDIT_PROLONG_SUM"], (-1.0 * (((((((data["REFUSED_AMT_ANNUITY_MEAN"]) < (
        data["REFUSED_AMT_ANNUITY_MAX"])) * 1.)) - ((((((data["REFUSED_AMT_ANNUITY_MEAN"]) < (data["REFUSED_AMT_ANNUITY_MAX"])) * 1.)) * ((-1.0 * ((data["ACTIVE_CNT_CREDIT_PROLONG_SUM"]))))))))))))
    v["i339"] = 0.098480 * np.tanh((((((((np.maximum(((((data["ORGANIZATION_TYPE_Trade__type_4"]) * (((data["ORGANIZATION_TYPE_Trade__type_4"]) + (3.141593)))))), ((
        ((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) / 2.0))))) < (np.minimum(((data["ORGANIZATION_TYPE_Trade__type_4"])), ((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]))))) * 1.)) * 2.0)) * 2.0))
    v["i340"] = 0.100000 * np.tanh(((((((((np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"] < -99998, data["AMT_CREDIT"], ((data["NAME_INCOME_TYPE_Student"]) * 2.0))) * (
        data["NEW_ANNUITY_TO_INCOME_RATIO"]))) * (data["NEW_ANNUITY_TO_INCOME_RATIO"]))) * (np.maximum(((0.0)), ((data["NEW_ANNUITY_TO_INCOME_RATIO"])))))) * 2.0))
    v["i341"] = 0.090000 * np.tanh(np.where(np.where(data["BURO_DAYS_CREDIT_ENDDATE_MEAN"] > 0, data["AMT_CREDIT"], (((data["EXT_SOURCE_3"]) > (data["AMT_CREDIT"])) * 1.))
                                            > 0, 0.318310, (-1.0 * (((((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) > ((((data["NAME_INCOME_TYPE_Student"]) > (data["AMT_CREDIT"])) * 1.))) * 1.))))))
    v["i342"] = 0.075000 * np.tanh(((np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"] < -99998, (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > ((2.53327202796936035))) * 1.), ((
        ((1.53511798381805420)) < (((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) - (0.318310)))) * 1.))) * ((((5.44602632522583008)) - (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])))))
    v["i343"] = 0.100000 * np.tanh(((np.minimum((((((((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) > (2.0)) * 1.)) / 2.0))), (((((8.44388580322265625)) - (np.maximum(
        ((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])), ((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]))))))))) * ((((8.44388580322265625)) + (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])))))
    v["i344"] = 0.050000 * np.tanh((-1.0 * (((((0.636620) < (np.where((((-1.0 * ((0.636620)))) + (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])) < 0,
                                                                      np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])), ((((data["AMT_CREDIT"]) / 2.0)))), (-1.0 * ((data["AMT_CREDIT"])))))) * 1.)))))
    v["i345"] = 0.099050 * np.tanh((((((-1.0 * (((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (-1.0))) > (
        ((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (-1.0))) > (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])) * 1.)) + ((5.21866846084594727))) / 2.0))) * 1.))))) * 2.0)) * 2.0))
    v["i346"] = 0.095400 * np.tanh((-1.0 * ((np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"] < -99998, data["NAME_INCOME_TYPE_Student"], np.where(data["AMT_ANNUITY"] > 0, data[
                                   "ORGANIZATION_TYPE_Trade__type_4"], np.where(data["BURO_DAYS_CREDIT_ENDDATE_MEAN"] < -99998, 1.570796, data["BURO_DAYS_CREDIT_ENDDATE_MEAN"])))))))
    v["i347"] = 0.092000 * np.tanh(np.where(data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"] < -99998, (((np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] > 0, data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"],
                                                                                                           (-1.0 * ((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) / 2.0)))))) < (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])) * 1.), (((3.141593) < (data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.)))
    v["i348"] = 0.099980 * np.tanh((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (((((5.0)) + (np.minimum(((3.141593)), ((
        (-1.0 * ((1.570796)))))))) / 2.0))) * 1.)) - ((((3.141593) < (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (-1.0)))) * 1.))))
    v["i349"] = 0.072006 * np.tanh((-1.0 * (((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) > (((data["AMT_ANNUITY"]) + (
        ((((((2.0)) + (0.318310)) / 2.0)) + (((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) * ((((0.318310) + (data["AMT_ANNUITY"])) / 2.0))))))))) * 1.)))))
    v["i350"] = 0.070695 * np.tanh(((np.minimum((((((0.636620) < (np.minimum(((data["AMT_ANNUITY"])), ((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]))))) * 1.))), ((
        (((np.maximum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) - (data["AMT_ANNUITY"])))))) < (1.570796)) * 1.))))) * 2.0))
    v["i351"] = 0.099010 * np.tanh(((3.0) * (((3.0) * ((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + ((((((3.141593) - (
        data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) > (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0))) * 1.)))) > (3.0)) * 1.))))))
    v["i352"] = 0.001700 * np.tanh(np.where(((np.maximum(((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) / 2.0))), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) + (-2.0))
                                            < 0, 0.0, np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"] < 0, ((data["AMT_ANNUITY"]) * ((9.0))), data["NEW_CREDIT_TO_ANNUITY_RATIO"])))
    v["i353"] = 0.037040 * np.tanh(np.where((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (-2.0)) / 2.0) < 0, np.where(((data["AMT_ANNUITY"]) * (data["NEW_ANNUITY_TO_INCOME_RATIO"]))
                                                                                                                   > 0, data["NAME_INCOME_TYPE_Maternity_leave"], data["ORGANIZATION_TYPE_Realtor"]), ((((((data["AMT_ANNUITY"]) * 2.0)) * 2.0)) * 2.0)))
    v["i354"] = 0.089980 * np.tanh(np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"] < 0, (((3.0) < ((((((-1.0 * ((data["ORGANIZATION_TYPE_Trade__type_4"])))) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) * 2.0)))
                                                                                        * 1.), np.where(data["NAME_HOUSING_TYPE_Municipal_apartment"] < 0, (-1.0 * ((data["ORGANIZATION_TYPE_Trade__type_4"]))), data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])))
    v["i355"] = 0.099952 * np.tanh(np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"] > 0, ((data["NAME_CONTRACT_TYPE_Revolving_loans"]) * ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < (
        data["NAME_CONTRACT_TYPE_Revolving_loans"])) * 1.))), (((((((data["NAME_CONTRACT_TYPE_Revolving_loans"]) / 2.0)) < (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])) * 1.)) * (data["NEW_ANNUITY_TO_INCOME_RATIO"]))))
    v["i356"] = 0.071280 * np.tanh(np.where(data["AMT_ANNUITY"] < 0, data["NAME_INCOME_TYPE_Maternity_leave"], np.where(data["NAME_HOUSING_TYPE_Municipal_apartment"] < 0, data[
                                   "NAME_INCOME_TYPE_Maternity_leave"], ((((((((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) * 2.0)) - (data["AMT_ANNUITY"]))) * ((14.04221153259277344)))) - (data["NAME_HOUSING_TYPE_Municipal_apartment"])))))
    v["i357"] = 0.067000 * np.tanh((((((2.0) < (((((data["AMT_ANNUITY"]) * ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])) * 1.)))) + (
        data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) * 1.)) * (((data["NEW_CREDIT_TO_INCOME_RATIO"]) + (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])))))
    v["i358"] = 0.098920 * np.tanh((((-1.0 * (((((((3.141593) + (((-1.0) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))) < (
        (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > ((((data["NAME_INCOME_TYPE_Maternity_leave"]) < (data["AMT_ANNUITY"])) * 1.))) * 1.))) * 1.))))) * 2.0))
    v["i359"] = 0.099730 * np.tanh((((((((data["AMT_ANNUITY"]) * 2.0)) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)) * (
        (((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)) > ((((data["REG_CITY_NOT_LIVE_CITY"]) < (((1.570796) + (((data["AMT_ANNUITY"]) * 2.0))))) * 1.))) * 1.))))
    v["i360"] = 0.096000 * np.tanh(((data["NAME_INCOME_TYPE_Maternity_leave"]) - (((((((((((((((data["AMT_ANNUITY"]) > (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])) * 1.))
                                                                                            > (data["AMT_INCOME_TOTAL"])) * 1.)) > (data["AMT_ANNUITY"])) * 1.)) + (data["AMT_INCOME_TOTAL"]))) > (1.570796)) * 1.))))
    v["i361"] = 0.098949 * np.tanh(((((((data["OCCUPATION_TYPE_Managers"]) * (np.maximum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), (((((-1.0) < (data["AMT_ANNUITY"])) * 1.)))))))
                                      * (np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])), (((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < (-1.0)) * 1.))))))) / 2.0))
    v["i362"] = 0.087000 * np.tanh(((((((((1.570796) > (data["AMT_ANNUITY"])) * 1.)) < ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (np.minimum(((np.where(
        data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"] > 0, (-1.0 * ((data["AMT_INCOME_TOTAL"]))), data["AMT_ANNUITY"]))), ((data["NAME_INCOME_TYPE_Student"]))))) / 2.0))) * 1.)) * 2.0))
    v["i363"] = 0.099560 * np.tanh(((((((((data["CC_AMT_RECEIVABLE_PRINCIPAL_MEAN"]) * 2.0)) * 2.0)) * ((((data["CC_AMT_RECEIVABLE_PRINCIPAL_MEAN"]) > (
        (((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) + ((((((data["CC_AMT_RECEIVABLE_PRINCIPAL_MEAN"]) + ((2.05832409858703613)))) + ((2.05832052230834961))) / 2.0))) / 2.0))) * 1.)))) * 2.0))
    v["i364"] = 0.099197 * np.tanh((-1.0 * ((np.where(data["CC_AMT_DRAWINGS_POS_CURRENT_VAR"] < 0, np.where(data["CC_CNT_DRAWINGS_POS_CURRENT_VAR"] < 0, np.where(
        data["CC_CNT_DRAWINGS_ATM_CURRENT_VAR"] < 0, (((data["NAME_INCOME_TYPE_Student"]) < (data["CC_CNT_DRAWINGS_ATM_CURRENT_VAR"])) * 1.), (5.0)), -2.0), data["CC_AMT_PAYMENT_CURRENT_MAX"])))))
    v["i365"] = 0.089978 * np.tanh(np.minimum((((((data["CC_AMT_RECEIVABLE_PRINCIPAL_MAX"]) < (data["CC_AMT_BALANCE_MAX"])) * 1.))), ((np.where(data["CC_AMT_BALANCE_MEAN"] > 0, data["CC_MONTHS_BALANCE_SUM"],
                                                                                                                                                (((((((((data["CC_AMT_BALANCE_MAX"]) < (data["CC_MONTHS_BALANCE_SUM"])) * 1.)) < (data["CC_AMT_RECEIVABLE_PRINCIPAL_MAX"])) * 1.)) < (data["CC_AMT_RECIVABLE_MAX"])) * 1.))))))
    v["i366"] = 0.099495 * np.tanh(((((((((0.318310) > ((-1.0 * ((((data["CC_AMT_BALANCE_MAX"]) * 2.0)))))) * 1.)) < (((data["CC_NAME_CONTRACT_STATUS_Refused_SUM"]) + (
        (((np.maximum(((data["AMT_INCOME_TOTAL"])), ((data["CC_AMT_TOTAL_RECEIVABLE_MAX"])))) + (data["CC_AMT_DRAWINGS_OTHER_CURRENT_SUM"])) / 2.0))))) * 1.)) * 2.0))
    v["i367"] = 0.099650 * np.tanh(((np.minimum(((data["CC_AMT_DRAWINGS_ATM_CURRENT_VAR"])), ((np.where(data["CC_AMT_PAYMENT_CURRENT_VAR"] > 0, 1.570796, ((data["CC_AMT_DRAWINGS_ATM_CURRENT_VAR"]) - (
        (((data["CC_AMT_PAYMENT_CURRENT_MEAN"]) > (((data["CC_AMT_PAYMENT_CURRENT_MAX"]) - (data["CC_AMT_PAYMENT_CURRENT_MEAN"])))) * 1.)))))))) - (data["CC_AMT_PAYMENT_CURRENT_MEAN"])))
    v["i368"] = 0.085001 * np.tanh(((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + ((((-1.0) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)))) * (np.where(data["CC_AMT_PAYMENT_CURRENT_VAR"] < -99998, data[
                                   "ORGANIZATION_TYPE_Industry__type_12"], (((data["CC_AMT_PAYMENT_CURRENT_VAR"]) > (((data["ORGANIZATION_TYPE_Industry__type_12"]) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) * 1.)))))
    v["i369"] = 0.081437 * np.tanh(np.where(data["CC_AMT_PAYMENT_CURRENT_MAX"] < -99998, 0.0, ((np.maximum(((((np.minimum(((0.0)), ((data["CC_AMT_DRAWINGS_ATM_CURRENT_VAR"])))) + (
        (((2.0) < (data["CC_CNT_DRAWINGS_ATM_CURRENT_VAR"])) * 1.))))), ((((data["CC_AMT_DRAWINGS_OTHER_CURRENT_VAR"]) * 2.0))))) * 2.0)))
    v["i370"] = 0.097400 * np.tanh(((((data["ORGANIZATION_TYPE_Industry__type_12"]) * (((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)) * 2.0)))) + (
        (((((data["ORGANIZATION_TYPE_Trade__type_4"]) > (((((data["ORGANIZATION_TYPE_Industry__type_12"]) * (np.tanh((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))) / 2.0))) * 1.)) / 2.0))))
    v["i371"] = 0.065150 * np.tanh(np.minimum(((np.minimum((((((9.97522354125976562)) * (((data["ORGANIZATION_TYPE_Trade__type_4"]) * (data["AMT_ANNUITY"])))))), ((
        ((data["ORGANIZATION_TYPE_Industry__type_12"]) * (data["REGION_POPULATION_RELATIVE"]))))))), ((((0.318310) * (np.tanh((data["REGION_POPULATION_RELATIVE"]))))))))
    v["i372"] = 0.099700 * np.tanh(np.minimum(((((data["ORGANIZATION_TYPE_Trade__type_4"]) * (-2.0)))), ((((((((-2.0) * 2.0)) * 2.0)) * (
        ((data["ORGANIZATION_TYPE_Industry__type_12"]) + (((-2.0) - ((-1.0 * ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))))))))))
    v["i373"] = 0.089000 * np.tanh((((np.where(np.where(data["AMT_INCOME_TOTAL"] < 0, data["AMT_ANNUITY"], data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < 0, data[
                                   "AMT_INCOME_TOTAL"], data["NEW_CREDIT_TO_ANNUITY_RATIO"])) > (np.maximum(((2.0)), ((((data["AMT_ANNUITY"]) * (3.0))))))) * 1.))
    v["i374"] = 0.010549 * np.tanh(((3.141593) * ((-1.0 * (((((3.141593) < (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, data["NEW_ANNUITY_TO_INCOME_RATIO"],
                                                                                     ((data["ORGANIZATION_TYPE_Transport__type_1"]) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (((3.0) / 2.0)))))))) * 1.)))))))
    v["i375"] = 0.070390 * np.tanh(((data["FONDKAPREMONT_MODE_not_specified"]) * (((data["FONDKAPREMONT_MODE_not_specified"]) * ((((data["AMT_INCOME_TOTAL"]) + (
        ((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (((((-1.0 * ((data["AMT_INCOME_TOTAL"])))) < (data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.))) * 1.)) + (data["NEW_ANNUITY_TO_INCOME_RATIO"])) / 2.0))) / 2.0))))))
    v["i376"] = 0.096994 * np.tanh((-1.0 * ((np.where(data["FONDKAPREMONT_MODE_reg_oper_spec_account"] > 0, (((-1.0) + (data["AMT_ANNUITY"])) / 2.0),
                                                      ((data["NAME_INCOME_TYPE_Student"]) * (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (data["NAME_INCOME_TYPE_Student"])))))))))
    v["i377"] = 0.005040 * np.tanh(((((data["ORGANIZATION_TYPE_Realtor"]) - (data["NAME_INCOME_TYPE_Student"]))) + (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (
        ((data["NAME_INCOME_TYPE_Maternity_leave"]) - (np.maximum(((((data["ORGANIZATION_TYPE_Realtor"]) - (data["NAME_INCOME_TYPE_Student"])))), ((data["ORGANIZATION_TYPE_Legal_Services"]))))))))))
    v["i378"] = 0.099709 * np.tanh(np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] < -99998, data["NEW_ANNUITY_TO_INCOME_RATIO"], (((np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] < 0, ((
        data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (1.570796)), data["AMT_ANNUITY"])) > (np.where(data["AMT_ANNUITY"] > 0, 3.141593, data["ORGANIZATION_TYPE_Trade__type_4"]))) * 1.)))
    v["i379"] = 0.049996 * np.tanh(np.where(data["AMT_INCOME_TOTAL"] < 0, ((data["ORGANIZATION_TYPE_Legal_Services"]) * 2.0), ((data["ORGANIZATION_TYPE_Legal_Services"]) - (((np.where(
        data["AMT_ANNUITY"] < 0, data["AMT_INCOME_TOTAL"], np.maximum(((data["ORGANIZATION_TYPE_Legal_Services"])), ((data["ORGANIZATION_TYPE_Trade__type_4"]))))) * (data["AMT_ANNUITY"]))))))
    v["i380"] = 0.040001 * np.tanh(np.minimum((((((((7.0)) * (data["ORGANIZATION_TYPE_Transport__type_1"]))) * (np.minimum((((-1.0 * ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))),
                                                                                                                           ((data["AMT_ANNUITY"]))))))), ((((((data["ORGANIZATION_TYPE_Transport__type_1"]) * (data["AMT_ANNUITY"]))) * (data["AMT_ANNUITY"]))))))
    v["i381"] = 0.082000 * np.tanh((((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (2.0)) * 1.)) / 2.0)) - ((((1.570796) < (((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (
        data["ORGANIZATION_TYPE_Transport__type_1"])) / 2.0)) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) / 2.0))) * 1.)))) * ((4.13047790527343750))))
    v["i382"] = 0.071302 * np.tanh((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (np.maximum(((data["AMT_ANNUITY"])), ((((np.maximum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])),
                                                                                                                                 (((((3.56457448005676270)) / 2.0))))) * (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (np.tanh((data["AMT_ANNUITY"]))))))))))) * 1.))
    v["i383"] = 0.0 * np.tanh(np.where(data["ORGANIZATION_TYPE_Industry__type_5"] < 0, (((-1.0 * ((data["ORGANIZATION_TYPE_Industry__type_5"])))) + (
        np.where((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (2.0)) * 1.) > 0, 3.141593, data["ORGANIZATION_TYPE_Realtor"]))), (-1.0 * ((2.0)))))
    v["i384"] = 0.000002 * np.tanh(((data["ORGANIZATION_TYPE_Realtor"]) - ((((((((((((data["ORGANIZATION_TYPE_Trade__type_4"]) + (data["NAME_TYPE_SUITE_Group_of_people"])) / 2.0)) * (
        data["NEW_ANNUITY_TO_INCOME_RATIO"]))) + ((((data["ORGANIZATION_TYPE_Trade__type_4"]) + (data["NAME_INCOME_TYPE_Student"])) / 2.0))) / 2.0)) + (data["ORGANIZATION_TYPE_Industry__type_13"])) / 2.0))))
    v["i385"] = 0.099700 * np.tanh((((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > ((((2.0)) + ((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (
        ((2.0) + (data["NAME_INCOME_TYPE_Maternity_leave"]))))) > ((0.11299613118171692))) * 1.))))) * 1.)) * 2.0)) * 2.0)) * 2.0))
    v["i386"] = 0.085100 * np.tanh(((((((((((2.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) / 2.0)) < (0.636620)) * 1.)) - (np.maximum(
        ((data["NAME_INCOME_TYPE_Student"])), ((((((((2.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) / 2.0)) < (0.318310)) * 1.))))))) / 2.0))
    v["i387"] = 0.074910 * np.tanh(((((((np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] < -99998, data["NEW_ANNUITY_TO_INCOME_RATIO"], (((data["NAME_INCOME_TYPE_Maternity_leave"]) + (
        (((data["NAME_INCOME_TYPE_Maternity_leave"]) + (((((2.60535907745361328)) < (data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"])) * 1.))) / 2.0))) / 2.0))) * 2.0)) * ((11.63154888153076172)))) * 2.0))
    v["i388"] = 0.099690 * np.tanh((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > ((2.0))) * 1.)) - (np.where(data["CLOSED_MONTHS_BALANCE_MIN_MIN"] > 0, data["CLOSED_MONTHS_BALANCE_MAX_MAX"], ((
        (data["CLOSED_MONTHS_BALANCE_MIN_MIN"]) > (np.where(data["CLOSED_MONTHS_BALANCE_MAX_MAX"] > 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], data["CLOSED_MONTHS_BALANCE_MAX_MAX"]))) * 1.)))))
    v["i389"] = 0.099990 * np.tanh(np.where(np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"] > 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], data["CC_NAME_CONTRACT_STATUS_Active_MEAN"]) < 0, (((np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] > 0, data[
                                   "CC_NAME_CONTRACT_STATUS_Active_MEAN"], data["NEW_CREDIT_TO_ANNUITY_RATIO"])) < (data["CC_NAME_CONTRACT_STATUS_Active_MEAN"])) * 1.), (((data["CC_NAME_CONTRACT_STATUS_Active_MEAN"]) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) / 2.0)))
    v["i390"] = 0.099600 * np.tanh(np.minimum((((((((data["CC_AMT_DRAWINGS_POS_CURRENT_VAR"]) > (np.tanh((((data["CC_AMT_PAYMENT_CURRENT_MEAN"]) / 2.0))))) * 1.)) + (
        ((data["CC_AMT_PAYMENT_CURRENT_VAR"]) + ((-1.0 * ((data["CC_AMT_PAYMENT_CURRENT_MEAN"]))))))))), (((((data["CC_AMT_DRAWINGS_ATM_CURRENT_VAR"]) > (data["CC_AMT_DRAWINGS_POS_CURRENT_VAR"])) * 1.)))))
    v["i391"] = 0.097600 * np.tanh((-1.0 * (((((data["CC_AMT_PAYMENT_CURRENT_MEAN"]) > (((((((-1.0 * ((np.maximum((((((data["CC_AMT_PAYMENT_CURRENT_MEAN"]) > (0.636620)) * 1.))), ((
        data["CC_AMT_DRAWINGS_OTHER_CURRENT_VAR"]))))))) < (data["CC_AMT_DRAWINGS_ATM_CURRENT_MAX"])) * 1.)) - (data["CC_AMT_DRAWINGS_ATM_CURRENT_MAX"])))) * 1.)))))
    v["i392"] = 0.092000 * np.tanh((((data["CC_AMT_DRAWINGS_ATM_CURRENT_VAR"]) > ((((((data["CC_AMT_PAYMENT_CURRENT_MEAN"]) + (np.where(((((-1.0 * ((data["CC_AMT_DRAWINGS_POS_CURRENT_VAR"])))) + (
        data["CC_AMT_DRAWINGS_ATM_CURRENT_VAR"])) / 2.0) > 0, data["CC_AMT_PAYMENT_CURRENT_MEAN"], data["CC_AMT_PAYMENT_CURRENT_VAR"]))) / 2.0)) / 2.0))) * 1.))
    v["i393"] = 0.066595 * np.tanh(np.where((((2.0) < (((data["CC_AMT_PAYMENT_CURRENT_MEAN"]) * 2.0))) * 1.) > 0, (-1.0 * ((((data["CC_AMT_DRAWINGS_POS_CURRENT_MEAN"]) * 2.0)))), np.where(
        data["CC_AMT_PAYMENT_CURRENT_MEAN"] < -99998, ((data["CC_AMT_DRAWINGS_POS_CURRENT_MEAN"]) - (data["CC_AMT_PAYMENT_CURRENT_MEAN"])), data["CC_AMT_DRAWINGS_POS_CURRENT_MEAN"])))
    v["i394"] = 0.100000 * np.tanh((-1.0 * (((((((((((((((((data["CC_AMT_DRAWINGS_POS_CURRENT_VAR"]) / 2.0)) / 2.0)) > ((-1.0 * ((data["CC_AMT_DRAWINGS_ATM_CURRENT_VAR"])))))
                                                      * 1.)) + (0.636620)) / 2.0)) + (data["CC_AMT_PAYMENT_CURRENT_MAX"]))) < (data["CC_AMT_PAYMENT_CURRENT_MEAN"])) * 1.)))))
    v["i395"] = 0.022210 * np.tanh(np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, ((((-1.0 * (((((data["ORGANIZATION_TYPE_Realtor"]) + (data["AMT_INCOME_TOTAL"])) / 2.0))))) < (
        data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.), (((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) > (np.maximum((((-1.0 * ((data["ORGANIZATION_TYPE_Industry__type_13"]))))), ((data["AMT_INCOME_TOTAL"]))))) * 1.)))
    v["i396"] = 0.000300 * np.tanh(np.where(data["ORGANIZATION_TYPE_Business_Entity_Type_1"] < 0, (((((((np.tanh((data["ORGANIZATION_TYPE_Business_Entity_Type_1"]))) < ((((data["ORGANIZATION_TYPE_Trade__type_4"]) + (
        ((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) * 2.0))) / 2.0))) * 1.)) / 2.0)) / 2.0), ((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) + (data["ORGANIZATION_TYPE_Business_Entity_Type_1"]))))
    v["i397"] = 0.085680 * np.tanh((((data["ORGANIZATION_TYPE_Business_Entity_Type_1"]) + (np.where(data["ORGANIZATION_TYPE_Business_Entity_Type_1"] > 0, data[
                                   "BURO_AMT_CREDIT_SUM_DEBT_MEAN"], np.where((((((1.570796) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) < ((-1.0 * ((0.318310))))) * 1.) > 0, -2.0, 0.318310)))) / 2.0))
    v["i398"] = 0.099020 * np.tanh(((np.minimum((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (2.0)) * 1.)) * 2.0))), ((
        (((2.0) + (((data["FLAG_OWN_REALTY"]) * (((data["AMT_ANNUITY"]) + (data["NEW_ANNUITY_TO_INCOME_RATIO"])))))) / 2.0))))) * 2.0))
    v["i399"] = 0.099771 * np.tanh(((((((13.78084564208984375)) * (np.where(data["CC_AMT_PAYMENT_CURRENT_VAR"] < 0, np.maximum(((data["NAME_INCOME_TYPE_Maternity_leave"])), (((((data["AMT_ANNUITY"]) > (
        data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])) * 1.)))), data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])))) < (np.minimum(((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])), ((data["FLAG_OWN_REALTY"]))))) * 1.))
    v["i400"] = 0.058997 * np.tanh(np.where((((data["CC_AMT_PAYMENT_CURRENT_VAR"]) > (data["CC_CNT_DRAWINGS_POS_CURRENT_VAR"])) * 1.) > 0, data["CC_CNT_DRAWINGS_POS_CURRENT_VAR"], np.where(
        (((((0.318310) * 2.0)) < (data["CC_CNT_DRAWINGS_POS_CURRENT_VAR"])) * 1.) > 0, data["CC_CNT_DRAWINGS_POS_CURRENT_VAR"], ((data["CC_AMT_DRAWINGS_OTHER_CURRENT_VAR"]) - (data["CC_CNT_DRAWINGS_POS_CURRENT_VAR"])))))
    v["i401"] = 0.051605 * np.tanh((((-1.0 * ((((np.where(data["CC_AMT_PAYMENT_CURRENT_MAX"] < -99998, ((data["CC_CNT_DRAWINGS_OTHER_CURRENT_VAR"]) - (
        data["CC_CNT_DRAWINGS_ATM_CURRENT_VAR"])), ((data["CC_AMT_PAYMENT_CURRENT_MAX"]) - (data["CC_AMT_PAYMENT_CURRENT_VAR"])))) * (data["CC_CNT_DRAWINGS_ATM_CURRENT_VAR"])))))) * 2.0))
    v["i402"] = 0.071032 * np.tanh(np.where(data["CC_AMT_DRAWINGS_ATM_CURRENT_VAR"] < 0, ((((data["CC_AMT_DRAWINGS_POS_CURRENT_VAR"]) - (data["CC_CNT_DRAWINGS_POS_CURRENT_VAR"]))) * (
        data["CC_AMT_DRAWINGS_POS_CURRENT_VAR"])), (((((data["CC_AMT_DRAWINGS_POS_CURRENT_VAR"]) < (data["CC_CNT_DRAWINGS_POS_CURRENT_VAR"])) * 1.)) - (data["CC_AMT_DRAWINGS_ATM_CURRENT_VAR"]))))
    v["i403"] = 0.099596 * np.tanh((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (np.where((((((data["NAME_FAMILY_STATUS_Civil_marriage"]) / 2.0)) > (
        ((((-1.0 * ((data["NEW_ANNUITY_TO_INCOME_RATIO"])))) + (data["FLAG_OWN_REALTY"])) / 2.0))) * 1.) > 0, (9.0), ((((10.0)) > (data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.)))) * 1.))
    v["i404"] = 0.009700 * np.tanh((((((((1.570796) < ((-1.0 * ((np.where((((data["AMT_ANNUITY"]) > (np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] > 0, 1.570796, data[
                                   "NEW_ANNUITY_TO_INCOME_RATIO"]))) * 1.) > 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], data["NEW_ANNUITY_TO_INCOME_RATIO"])))))) * 1.)) * 2.0)) * 2.0))
    v["i405"] = 0.012100 * np.tanh(((np.where(data["ORGANIZATION_TYPE_Mobile"] > 0, ((data["NAME_INCOME_TYPE_Maternity_leave"]) - (data["AMT_ANNUITY"])), ((
        (data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (((3.0) + (np.minimum(((data["AMT_INCOME_TOTAL"])), ((data["ORGANIZATION_TYPE_Mobile"]))))))) * 1.))) * ((6.0))))
    v["i406"] = 0.012404 * np.tanh((-1.0 * ((((data["WALLSMATERIAL_MODE_Mixed"]) * (((data["WALLSMATERIAL_MODE_Mixed"]) * ((((-1.0) + (
        ((data["WALLSMATERIAL_MODE_Mixed"]) * ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (data["AMT_ANNUITY"])) / 2.0))))) / 2.0)))))))))
    v["i407"] = 0.097502 * np.tanh(np.where(((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["AMT_ANNUITY"])) > 0, ((
        (((((((data["AMT_ANNUITY"]) > (3.141593)) * 1.)) * (data["AMT_ANNUITY"]))) - (data["ORGANIZATION_TYPE_Realtor"]))) * 2.0), data["ORGANIZATION_TYPE_Realtor"]))
    v["i408"] = 0.099970 * np.tanh(np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] < -99998, data["NEW_CREDIT_TO_ANNUITY_RATIO"], (((((np.maximum((((2.58687329292297363))), ((
        (((((((1.570796) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)) * (3.141593))) * 2.0))))) < (data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.)) * 2.0)))
    v["i409"] = 0.084550 * np.tanh((-1.0 * ((np.where((((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) > ((9.0))) * 1.) > 0, (10.0), np.where(
        (((data["AMT_INCOME_TOTAL"]) + (-2.0)) / 2.0) > 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], (((data["ORGANIZATION_TYPE_Industry__type_13"]) > ((10.0))) * 1.)))))))
    v["i410"] = 0.037702 * np.tanh(((data["ORGANIZATION_TYPE_Realtor"]) + (((data["ORGANIZATION_TYPE_Realtor"]) + ((((np.where((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (
        data["ORGANIZATION_TYPE_Realtor"])) * 1.) > 0, data["ORGANIZATION_TYPE_Realtor"], ((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) + (1.0)))) > (3.0)) * 1.))))))
    v["i411"] = 0.085500 * np.tanh((((-1.0 * (((((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) > (np.where(((1.570796) + (data["AMT_ANNUITY"])) < 0, ((
        data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) + (1.0)))), (8.41182804107666016)))) * 1.))))) * 2.0))
    v["i412"] = 0.011900 * np.tanh((((((((1.570796) < (((((((data["ORGANIZATION_TYPE_Industry__type_5"]) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) / 2.0)) + (
        data["NEW_CREDIT_TO_ANNUITY_RATIO"])) / 2.0))) * 1.)) * (((data["ORGANIZATION_TYPE_Mobile"]) - (((1.570796) * 2.0)))))) * 2.0))
    v["i413"] = 0.097005 * np.tanh(((np.maximum(((np.maximum(((data["ORGANIZATION_TYPE_Trade__type_4"])), ((data["ORGANIZATION_TYPE_Mobile"]))))), ((
        data["NAME_INCOME_TYPE_Student"])))) * (((np.where(-1.0 > 0, data["AMT_ANNUITY"], np.tanh((-1.0)))) - (data["AMT_ANNUITY"])))))
    v["i414"] = 0.029970 * np.tanh(((((((np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"] > 0, -2.0, data["ORGANIZATION_TYPE_Trade__type_4"])) - ((-1.0 * ((data["NAME_TYPE_SUITE_Group_of_people"])))))) * (
        data["ORGANIZATION_TYPE_Mobile"]))) + (np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"] < -99998, data["NAME_TYPE_SUITE_Group_of_people"], data["ORGANIZATION_TYPE_Mobile"]))))
    v["i415"] = 0.098026 * np.tanh(((np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"] < -99998, ((((-1.0 * ((((data["AMT_INCOME_TOTAL"]) * (data["AMT_INCOME_TOTAL"])))))) > (np.where(
        data["AMT_INCOME_TOTAL"] > 0, data["ORGANIZATION_TYPE_Trade__type_7"], data["ORGANIZATION_TYPE_Agriculture"]))) * 1.), data["AMT_INCOME_TOTAL"])) * (data["ORGANIZATION_TYPE_Trade__type_7"])))
    v["i416"] = 0.074000 * np.tanh((-1.0 * (((((3.141593) < ((((np.where(data["AMT_INCOME_TOTAL"] < 0, ((3.141593) * 2.0), data["BURO_CNT_CREDIT_PROLONG_SUM"])) + (
        (((data["BURO_CREDIT_TYPE_Another_type_of_loan_MEAN"]) + ((((3.52182722091674805)) * (data["BURO_CREDIT_TYPE_Car_loan_MEAN"])))) / 2.0))) / 2.0))) * 1.)))))
    v["i417"] = 0.029946 * np.tanh(np.where(data["ORGANIZATION_TYPE_Mobile"] < 0, ((((data["ORGANIZATION_TYPE_Agriculture"]) * ((((((data["ORGANIZATION_TYPE_Business_Entity_Type_3"]) * (
        ((data["ORGANIZATION_TYPE_Business_Entity_Type_3"]) * (data["CODE_GENDER"]))))) + (data["ORGANIZATION_TYPE_Mobile"])) / 2.0)))) - (data["NAME_INCOME_TYPE_Student"])), data["ORGANIZATION_TYPE_Mobile"]))
    v["i418"] = 0.090020 * np.tanh(np.where(np.where(data["NEW_INC_PER_CHLD"] < 0, data["AMT_INCOME_TOTAL"], data["ORGANIZATION_TYPE_Agriculture"]) < 0, ((data["WALLSMATERIAL_MODE_Mixed"]) * (
        (-1.0 * ((data["ORGANIZATION_TYPE_Industry__type_11"]))))), np.where(data["ORGANIZATION_TYPE_Agriculture"] < 0, data["AMT_INCOME_TOTAL"], (-1.0 * ((data["ORGANIZATION_TYPE_Agriculture"]))))))
    v["i419"] = 0.100000 * np.tanh(((np.where(data["ACTIVE_CNT_CREDIT_PROLONG_SUM"] < 0, data["NAME_INCOME_TYPE_Maternity_leave"], ((np.where(
        ((data["AMT_INCOME_TOTAL"]) - (data["NEW_INC_PER_CHLD"])) < 0, data["NEW_INC_PER_CHLD"], (4.0))) - (((2.0) / 2.0))))) * ((6.28400230407714844))))
    v["i420"] = 0.099760 * np.tanh(np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"])), (((((((data["NEW_INC_PER_CHLD"]) + ((((2.0) + (data["AMT_INCOME_TOTAL"])) / 2.0))))
                                                                                                  < (((((((data["AMT_INCOME_TOTAL"]) < (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"])) * 1.)) < (data["NEW_INC_PER_CHLD"])) * 1.))) * 1.)))))
    v["i421"] = 0.092981 * np.tanh((((((((data["CNT_CHILDREN"]) > ((((((((np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"])), ((data["ORGANIZATION_TYPE_Trade__type_4"]))))
                                                                         < ((((((data["CNT_CHILDREN"]) > (data["ORGANIZATION_TYPE_Trade__type_7"])) * 1.)) / 2.0))) * 1.)) * 2.0)) * 2.0))) * 1.)) * 2.0)) * 2.0))
    v["i422"] = 0.099500 * np.tanh(((np.minimum(((np.where(data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"] < 0, -2.0, data["NEW_INC_PER_CHLD"]))), ((data["AMT_INCOME_TOTAL"])))) * (
        (((np.maximum(((data["ACTIVE_CREDIT_DAY_OVERDUE_MEAN"])), (((((0.636620) + (data["NEW_INC_PER_CHLD"])) / 2.0))))) < (data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"])) * 1.))))
    v["i423"] = 0.001995 * np.tanh((((0.0)) - (np.where(data["NEW_DOC_IND_KURT"] > 0, data["ORGANIZATION_TYPE_Trade__type_4"], ((((((data["ORGANIZATION_TYPE_Trade__type_7"]) < (
        data["AMT_INCOME_TOTAL"])) * 1.)) + ((((data["ORGANIZATION_TYPE_Trade__type_7"]) + ((((data["NEW_DOC_IND_KURT"]) + (data["AMT_INCOME_TOTAL"])) / 2.0))) / 2.0))) / 2.0)))))
    v["i424"] = 0.096014 * np.tanh((((((-1.0 * ((np.maximum(((data["NAME_EDUCATION_TYPE_Academic_degree"])), (((-1.0 * ((data["NAME_TYPE_SUITE_Group_of_people"])))))))))) * (
        (-1.0 * ((data["WEEKDAY_APPR_PROCESS_START_WEDNESDAY"])))))) - (((data["ORGANIZATION_TYPE_Trade__type_4"]) + ((((data["ORGANIZATION_TYPE_Trade__type_7"]) > (0.636620)) * 1.))))))
    v["i425"] = 0.097301 * np.tanh(((data["FONDKAPREMONT_MODE_not_specified"]) * (((((data["NEW_DOC_IND_KURT"]) * ((((data["NEW_DOC_IND_KURT"]) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.))))
                                                                                   * (np.maximum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), (((((data["AMT_INCOME_TOTAL"]) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)))))))))
    v["i426"] = 0.089008 * np.tanh(np.where((((data["ORGANIZATION_TYPE_Trade__type_7"]) + (data["WEEKDAY_APPR_PROCESS_START_SUNDAY"])) / 2.0) > 0, data["FLAG_DOCUMENT_3"], ((
        data["ORGANIZATION_TYPE_Security_Ministries"]) * (np.where(data["NAME_TYPE_SUITE_Group_of_people"] > 0, data["FLAG_DOCUMENT_3"], ((data["ORGANIZATION_TYPE_Security_Ministries"]) * (data["NEW_INC_PER_CHLD"])))))))
    v["i427"] = 0.099750 * np.tanh(np.where(((data["EXT_SOURCE_2"]) - (data["NAME_INCOME_TYPE_Student"])) < 0, data["AMT_INCOME_TOTAL"], ((
        (((-1.0 * ((data["EXT_SOURCE_2"])))) * (((data["EXT_SOURCE_2"]) - ((((data["EXT_SOURCE_2"]) < (1.0)) * 1.)))))) * 2.0)))
    v["i428"] = 0.099890 * np.tanh((((((np.minimum(((data["NEW_EMPLOY_TO_BIRTH_RATIO"])), ((data["DAYS_EMPLOYED"])))) > (np.where(((data["NEW_EMPLOY_TO_BIRTH_RATIO"]) * (
        data["DAYS_EMPLOYED"])) < 0, ((data["NAME_INCOME_TYPE_Student"]) * 2.0), np.tanh((np.tanh((data["NEW_EMPLOY_TO_BIRTH_RATIO"]))))))) * 1.)) * 2.0))
    v["i429"] = 0.091597 * np.tanh(((((((((data["NEW_CAR_TO_EMPLOY_RATIO"]) - (data["DAYS_EMPLOYED"]))) > ((((data["NEW_EMPLOY_TO_BIRTH_RATIO"]) > (
        (((data["NEW_EMPLOY_TO_BIRTH_RATIO"]) > (0.636620)) * 1.))) * 1.))) * 1.)) > ((((1.570796) < (data["NEW_EMPLOY_TO_BIRTH_RATIO"])) * 1.))) * 1.))
    v["i430"] = 0.090401 * np.tanh(np.where(data["DAYS_EMPLOYED"] < -99998, ((3.141593) - (3.0)), np.where(data["FLAG_OWN_CAR"] < 0, ((
        (3.141593) < (data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"])) * 1.), ((3.0) * (data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"])))))
    v["i431"] = 0.099920 * np.tanh(((((((data["REFUSED_DAYS_DECISION_MEAN"]) * (np.where(data["REFUSED_CNT_PAYMENT_SUM"] < 0, 0.0, ((np.where(
        data["REFUSED_AMT_APPLICATION_MEAN"] > 0, data["REFUSED_CNT_PAYMENT_SUM"], 0.0)) - (3.0)))))) * (data["REFUSED_AMT_APPLICATION_MEAN"]))) * 2.0))
    v["i432"] = 0.091453 * np.tanh((((((2.0) < (np.where(data["NAME_INCOME_TYPE_State_servant"] < 0, (-1.0 * (
        (((data["EXT_SOURCE_2"]) - ((-1.0 * ((0.318310)))))))), data["REGION_POPULATION_RELATIVE"]))) * 1.)) * 2.0))
    v["i433"] = 0.095979 * np.tanh(np.where(((data["NAME_FAMILY_STATUS_Civil_marriage"]) - (((data["NEW_EXT_SOURCES_MEAN"]) + ((2.0))))) < 0, (((np.where(data["NEW_EXT_SOURCES_MEAN"] < 0, data[
                                   "NEW_EXT_SOURCES_MEAN"], data["NAME_FAMILY_STATUS_Civil_marriage"])) > (data["ORGANIZATION_TYPE_Military"])) * 1.), data["NAME_INCOME_TYPE_State_servant"]))
    v["i434"] = 0.054620 * np.tanh(np.where(data["CC_AMT_PAYMENT_CURRENT_VAR"] < -99998, ((np.tanh((((data["NEW_EXT_SOURCES_MEAN"]) * 2.0)))) * (np.minimum(
        ((data["REGION_RATING_CLIENT"])), ((0.318310))))), np.minimum(((data["NEW_EXT_SOURCES_MEAN"])), ((((data["REGION_RATING_CLIENT"]) + (data["REGION_POPULATION_RELATIVE"])))))))
    v["i435"] = 0.011396 * np.tanh(((((((np.maximum(((data["CC_AMT_RECIVABLE_MEAN"])), ((
        data["CC_AMT_BALANCE_MEAN"])))) - (data["CC_AMT_INST_MIN_REGULARITY_MEAN"]))) * 2.0)) * 2.0))
    v["i436"] = 0.090200 * np.tanh((((((((data["CC_AMT_RECIVABLE_MEAN"]) > ((((((data["CC_AMT_RECEIVABLE_PRINCIPAL_MEAN"]) > (
        data["CC_AMT_TOTAL_RECEIVABLE_MEAN"])) * 1.)) + (((((0.10667683929204941)) < (data["CC_AMT_TOTAL_RECEIVABLE_MEAN"])) * 1.))))) * 1.)) * 2.0)) * 2.0))
    v["i437"] = 0.088430 * np.tanh(np.where(np.where(data["NAME_TYPE_SUITE_Group_of_people"] < 0, data["ORGANIZATION_TYPE_Trade__type_7"], data["NEW_EXT_SOURCES_MEAN"])
                                            < 0, (((data["NEW_EXT_SOURCES_MEAN"]) > (1.570796)) * 1.), np.where(data["NEW_EXT_SOURCES_MEAN"] < 0, data["ORGANIZATION_TYPE_Trade__type_7"], -2.0)))
    v["i438"] = 0.013965 * np.tanh(np.where(data["NEW_EXT_SOURCES_MEAN"] > 0, ((np.tanh((np.tanh((data["NEW_ANNUITY_TO_INCOME_RATIO"]))))) / 2.0),
                                            (-1.0 * ((np.maximum(((data["ORGANIZATION_TYPE_Military"])), ((((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) / 2.0)) / 2.0)) / 2.0)))))))))
    v["i439"] = 0.097690 * np.tanh((-1.0 * (((((-1.0) > (np.maximum((((((((data["NEW_EXT_SOURCES_MEAN"]) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) + (
        ((-1.0) * 2.0))) / 2.0))), ((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)) + (0.636620))))))) * 1.)))))
    v["i440"] = 0.099999 * np.tanh((((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > ((3.86333322525024414))) * 1.)) * (data["AMT_CREDIT"]))) - (
        ((((3.86333322525024414)) < (np.maximum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((((data["AMT_CREDIT"]) * (data["NEW_ANNUITY_TO_INCOME_RATIO"]))))))) * 1.))))
    v["i441"] = 0.099940 * np.tanh(np.maximum((((((np.tanh((1.570796))) < (((data["NEW_CAR_TO_EMPLOY_RATIO"]) - ((((data["ORGANIZATION_TYPE_Industry__type_5"]) + (
        data["NEW_EXT_SOURCES_MEAN"])) / 2.0))))) * 1.))), (((((((data["AMT_CREDIT"]) < (data["NEW_EXT_SOURCES_MEAN"])) * 1.)) * (data["AMT_CREDIT"]))))))
    v["i442"] = 0.020019 * np.tanh((((np.where(((-1.0) + (data["AMT_CREDIT"])) < 0, data["OWN_CAR_AGE"], ((np.where(
        data["NEW_EXT_SOURCES_MEAN"] < 0, ((-1.0) + (data["AMT_CREDIT"])), -1.0)) * 2.0))) > ((1.32165348529815674))) * 1.))
    v["i443"] = 0.090000 * np.tanh(((((np.where((-1.0 * ((((0.636620) + (((data["AMT_CREDIT"]) / 2.0)))))) < 0, ((
        data["ORGANIZATION_TYPE_Trade__type_5"]) * (((0.636620) + (data["FLAG_OWN_CAR"])))), data["NEW_CAR_TO_BIRTH_RATIO"])) * 2.0)) * 2.0))
    v["i444"] = 0.069500 * np.tanh((((((((((3.0) - (data["AMT_CREDIT"]))) < ((((np.where(data["NEW_EXT_SOURCES_MEAN"]
                                                                                         > 0, 0.318310, ((3.0) - (0.318310)))) + (data["BURO_STATUS_5_MEAN_MEAN"])) / 2.0))) * 1.)) * 2.0)) * 2.0))
    v["i445"] = 0.099749 * np.tanh(np.where(((data["NEW_CAR_TO_EMPLOY_RATIO"]) * ((((data["AMT_CREDIT"]) < (data["NEW_EXT_SOURCES_MEAN"])) * 1.))) > 0, data["NEW_EXT_SOURCES_MEAN"], ((
        data["AMT_CREDIT"]) * (np.minimum(((data["NAME_EDUCATION_TYPE_Lower_secondary"])), ((((data["NAME_INCOME_TYPE_Student"]) - (data["NAME_EDUCATION_TYPE_Lower_secondary"])))))))))
    v["i446"] = 0.033999 * np.tanh((-1.0 * (((((((data["NEW_CAR_TO_BIRTH_RATIO"]) > (data["NEW_CAR_TO_EMPLOY_RATIO"])) * 1.)) * (np.maximum(((np.where(data["OWN_CAR_AGE"] < 0, ((
        data["NEW_CAR_TO_BIRTH_RATIO"]) * (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])), (-1.0 * ((data["NEW_CREDIT_TO_INCOME_RATIO"])))))), ((data["NEW_EXT_SOURCES_MEAN"])))))))))
    v["i447"] = 0.099649 * np.tanh(((((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) - (-2.0))) + (
        ((3.0) / 2.0)))) * ((((data["NEW_EXT_SOURCES_MEAN"]) > ((((2.36968207359313965)) / 2.0))) * 1.))))
    v["i448"] = 0.099899 * np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"] > 0, np.where(data["ACTIVE_AMT_CREDIT_SUM_SUM"] < 0, data["ACTIVE_DAYS_CREDIT_MEAN"], data[
                                   "ACTIVE_AMT_CREDIT_SUM_SUM"]), ((((((3.0) < (((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) * 2.0))) * 1.)) > ((((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) > (2.0)) * 1.))) * 1.)))
    v["i449"] = 0.049990 * np.tanh(((((((data["ACTIVE_CNT_CREDIT_PROLONG_SUM"]) * 2.0)) * 2.0)) * (np.where(data["ACTIVE_CNT_CREDIT_PROLONG_SUM"] < 0, (((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]) < (
        data["ACTIVE_AMT_CREDIT_SUM_SUM"])) * 1.), ((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]) - ((((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) > (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"])) * 1.)))))))
    v["i450"] = 0.099984 * np.tanh((((data["NEW_EXT_SOURCES_MEAN"]) > (((((((3.0) + (data["ORGANIZATION_TYPE_Industry__type_5"]))) + (data["ORGANIZATION_TYPE_Trade__type_4"]))) - (
        np.where(((data["NEW_EXT_SOURCES_MEAN"]) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) > 0, data["NEW_CREDIT_TO_INCOME_RATIO"], data["ORGANIZATION_TYPE_Industry__type_1"]))))) * 1.))
    v["i451"] = 0.097200 * np.tanh(np.where(data["NEW_EXT_SOURCES_MEAN"] < 0, data["ORGANIZATION_TYPE_Advertising"], np.where(((data["AMT_ANNUITY"]) * (
        data["NEW_ANNUITY_TO_INCOME_RATIO"])) < 0, -1.0, (((data["NEW_EXT_SOURCES_MEAN"]) < ((((data["ORGANIZATION_TYPE_Industry__type_5"]) + (data["AMT_ANNUITY"])) / 2.0))) * 1.))))
    v["i452"] = 0.099052 * np.tanh((-1.0 * ((np.where(np.where(data["BURO_CNT_CREDIT_PROLONG_SUM"] > 0, data["NEW_CAR_TO_EMPLOY_RATIO"], data["NEW_EXT_SOURCES_MEAN"]) < -99998, 1.570796, np.where(
        data["BURO_CREDIT_TYPE_Another_type_of_loan_MEAN"] < 0, np.where(data["BURO_CNT_CREDIT_PROLONG_SUM"] > 0, -2.0, data["ORGANIZATION_TYPE_Industry__type_5"]), 3.141593))))))
    v["i453"] = 0.056500 * np.tanh((((((((data["NEW_CAR_TO_BIRTH_RATIO"]) > ((((((2.0) * ((-1.0 * ((data["ORGANIZATION_TYPE_Trade__type_4"])))))) < (
        (-1.0 * (((((data["OWN_CAR_AGE"]) + (data["NEW_CAR_TO_EMPLOY_RATIO"])) / 2.0)))))) * 1.))) * 1.)) * 2.0)) * 2.0))
    v["i454"] = 0.099945 * np.tanh(np.where(((data["FLAG_EMP_PHONE"]) + (data["NEW_CAR_TO_EMPLOY_RATIO"])) < 0, (((1.570796) < (data["OWN_CAR_AGE"])) * 1.), np.where(
        (((data["FLAG_EMP_PHONE"]) > (data["NEW_CAR_TO_BIRTH_RATIO"])) * 1.) > 0, data["NEW_CAR_TO_EMPLOY_RATIO"], data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"])))
    v["i455"] = 0.098400 * np.tanh(((((((((0.636620) > (data["OWN_CAR_AGE"])) * 1.)) > (np.tanh(((((data["DAYS_EMPLOYED"]) + (
        data["DAYS_EMPLOYED"])) / 2.0))))) * 1.)) * ((-1.0 * (((((data["DAYS_EMPLOYED"]) > ((-1.0 * ((data["OWN_CAR_AGE"]))))) * 1.)))))))
    v["i456"] = 0.028998 * np.tanh((-1.0 * ((((data["ORGANIZATION_TYPE_Trade__type_4"]) + (np.where(data["ORGANIZATION_TYPE_Trade__type_4"] < 0, ((data["NAME_INCOME_TYPE_Student"]) + (
        ((np.minimum(((data["ORGANIZATION_TYPE_Trade__type_4"])), ((data["ORGANIZATION_TYPE_Security_Ministries"])))) + (data["NAME_INCOME_TYPE_Student"])))), 2.0)))))))
    v["i457"] = 0.069700 * np.tanh((((data["ORGANIZATION_TYPE_Transport__type_3"]) > (np.where(((((((data["AMT_ANNUITY"]) + ((((data["ORGANIZATION_TYPE_Realtor"]) + (data["AMT_ANNUITY"])) / 2.0))) / 2.0)) + (
        (((data["ORGANIZATION_TYPE_Transport__type_3"]) + (data["NEW_ANNUITY_TO_INCOME_RATIO"])) / 2.0))) / 2.0) < 0, data["ORGANIZATION_TYPE_Transport__type_3"], data["AMT_ANNUITY"]))) * 1.))
    v["i458"] = 0.099350 * np.tanh(np.maximum((((-1.0 * ((data["NAME_INCOME_TYPE_Student"]))))), ((np.where(np.where(data["NEW_CAR_TO_EMPLOY_RATIO"]
                                                                                                                     < -99998, 0.318310, data["NEW_CAR_TO_EMPLOY_RATIO"]) > 0, data["OWN_CAR_AGE"], (-1.0 * ((((data["OWN_CAR_AGE"]) * 2.0)))))))))
    v["i459"] = 0.074200 * np.tanh((((data["NEW_CREDIT_TO_INCOME_RATIO"]) > (np.where(data["NEW_INC_PER_CHLD"] < 0, (8.19191551208496094), ((
        data["ORGANIZATION_TYPE_Business_Entity_Type_3"]) + (np.where((((data["NEW_INC_PER_CHLD"]) + (-1.0)) / 2.0) < 0, 2.0, data["NAME_INCOME_TYPE_Student"])))))) * 1.))
    v["i460"] = 0.093970 * np.tanh((((-1.0 * (((((3.0) < (np.where(np.where(data["AMT_INCOME_TOTAL"] > 0, data["ORGANIZATION_TYPE_Business_Entity_Type_3"], 0.636620) > 0, data[
                                   "NEW_CREDIT_TO_INCOME_RATIO"], ((data["AMT_CREDIT"]) - (data["ORGANIZATION_TYPE_Business_Entity_Type_3"]))))) * 1.))))) * (3.141593)))
    v["i461"] = 0.080000 * np.tanh((((-2.0) > ((((((np.where(data["AMT_ANNUITY"] > 0, ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * ((((-1.0 * ((data["AMT_INCOME_TOTAL"])))) * 2.0))),
                                                             (-1.0 * ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))) * 2.0)) + (data["ORGANIZATION_TYPE_Business_Entity_Type_3"])) / 2.0))) * 1.))
    v["i462"] = 0.099397 * np.tanh((-1.0 * (((((((((((3.141593) - (((data["REGION_RATING_CLIENT"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))) < (
        (((((3.141593) - (data["REGION_RATING_CLIENT"]))) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.))) * 1.)) * 2.0)) * 2.0)))))
    v["i463"] = 0.092303 * np.tanh(((((np.where((((((data["NAME_INCOME_TYPE_Student"]) * (data["ORGANIZATION_TYPE_Industry__type_13"]))) < (data["ORGANIZATION_TYPE_Security_Ministries"])) * 1.) > 0, data[
                                   "FONDKAPREMONT_MODE_reg_oper_account"], ((data["FONDKAPREMONT_MODE_reg_oper_account"]) * ((-1.0 * ((data["ORGANIZATION_TYPE_Industry__type_13"]))))))) * 2.0)) * 2.0))
    v["i464"] = 0.018040 * np.tanh((((((np.where(data["FLOORSMAX_MODE"] < 0, data["FLOORSMAX_MODE"], data["NONLIVINGAREA_MEDI"])) < (
        ((1.0) - ((((2.44985151290893555)) - (data["LIVINGAPARTMENTS_AVG"])))))) * 1.)) - ((((data["FLOORSMAX_MODE"]) > (1.0)) * 1.))))
    v["i465"] = 0.070000 * np.tanh((((((data["LIVINGAPARTMENTS_MODE"]) < (np.where(data["LIVINGAPARTMENTS_MODE"] > 0, 0.318310, data[
                                   "LIVINGAPARTMENTS_AVG"]))) * 1.)) * (((((-1.0) + (((data["ORGANIZATION_TYPE_Business_Entity_Type_3"]) * 2.0)))) * 2.0))))
    v["i466"] = 0.094990 * np.tanh(((((((((data["CC_NAME_CONTRACT_STATUS_Refused_MAX"]) + ((((data["CC_CNT_DRAWINGS_CURRENT_MEAN"]) > (((((np.minimum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])),
                                                                                                                                                      ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) / 2.0)) / 2.0))) * 1.))) / 2.0)) > ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < (data["AMT_ANNUITY"])) * 1.))) * 1.)) * 2.0))
    v["i467"] = 0.069960 * np.tanh(np.where(data["CC_AMT_PAYMENT_CURRENT_VAR"] < -99998, (((data["CC_CNT_DRAWINGS_POS_CURRENT_VAR"]) < (data["CC_AMT_PAYMENT_CURRENT_MEAN"])) * 1.), (-1.0 * (
        ((((data["CC_AMT_PAYMENT_CURRENT_MEAN"]) > (np.where(data["CC_AMT_PAYMENT_CURRENT_MAX"] < 0, (((data["CC_AMT_PAYMENT_CURRENT_MEAN"]) > (0.636620)) * 1.), 0.318310))) * 1.))))))
    v["i468"] = 0.080521 * np.tanh(np.tanh((np.maximum(((np.where(data["OCCUPATION_TYPE_Cooking_staff"] > 0, data["NAME_EDUCATION_TYPE_Higher_education"], data["ACTIVE_DAYS_CREDIT_VAR"]))), ((
        (((((((data["NAME_EDUCATION_TYPE_Higher_education"]) + ((((data["OCCUPATION_TYPE_Cooking_staff"]) < (data["ACTIVE_DAYS_CREDIT_VAR"])) * 1.))) / 2.0)) * (data["AMT_ANNUITY"]))) / 2.0)))))))
    v["i469"] = 0.094998 * np.tanh(((data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"]) * (((((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]) + (np.maximum(((data["ACTIVE_CNT_CREDIT_PROLONG_SUM"])), ((
        (((data["ACTIVE_MONTHS_BALANCE_SIZE_SUM"]) > (1.0)) * 1.))))))) * (((data["ACTIVE_MONTHS_BALANCE_SIZE_SUM"]) - (data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"])))))))
    v["i470"] = 0.094799 * np.tanh(((np.minimum(((np.where(data["ACTIVE_DAYS_CREDIT_MEAN"] < -99998, 0.0, (((((((data["ACTIVE_DAYS_CREDIT_MEAN"]) < (-1.0)) * 1.))
                                                                                                             * 2.0)) + (data["ACTIVE_DAYS_CREDIT_MEAN"]))))), ((((((0.0)) < (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"])) * 1.))))) / 2.0))
    v["i471"] = 0.095500 * np.tanh((((((((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]) < (data["ACTIVE_AMT_CREDIT_SUM_SUM"])) * 1.)) * (((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) * (
        ((np.where(data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"] < -99998, data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"], 0.636620)) * (data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]))))))) * 2.0)) * 2.0))
    v["i472"] = 0.085900 * np.tanh((((((((data["NEW_CREDIT_TO_INCOME_RATIO"]) / 2.0)) - (np.where(((data["NEW_CREDIT_TO_INCOME_RATIO"]) * (data["AMT_INCOME_TOTAL"]))
                                                                                                  < 0, ((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["AMT_INCOME_TOTAL"])), (-1.0 * ((data["NEW_ANNUITY_TO_INCOME_RATIO"]))))))) > (3.141593)) * 1.))
    v["i473"] = 0.099010 * np.tanh((((((data["AMT_INCOME_TOTAL"]) > (np.where((((3.0) < (data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"])) * 1.) > 0, ((
        data["AMT_INCOME_TOTAL"]) * (data["AMT_INCOME_TOTAL"])), ((3.0) / 2.0)))) * 1.)) * (data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"])))
    v["i474"] = 0.020201 * np.tanh((((((-1.0 * ((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)) - (data["OCCUPATION_TYPE_Cooking_staff"]))) - (
        np.tanh((((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) / 2.0))))))))) * (data["ORGANIZATION_TYPE_University"]))) * (data["AMT_ANNUITY"])))
    v["i475"] = 0.099490 * np.tanh((((((-1.0) > (((((data["AMT_ANNUITY"]) + (np.where(data["AMT_INCOME_TOTAL"] < 0, np.where(data["AMT_ANNUITY"] < 0, data[
                                   "AMT_INCOME_TOTAL"], ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)), data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) / 2.0))) * 1.)) * 2.0))
    v["i476"] = 0.097001 * np.tanh(np.where(data["INSTAL_DAYS_ENTRY_PAYMENT_STD"] < -99998, 0.318310, (-1.0 * ((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]
                                                                                                                         > 0, (((1.0) < (data["INSTAL_DAYS_ENTRY_PAYMENT_STD"])) * 1.), ((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]) - ((((-1.0 * ((1.0)))) / 2.0)))))))))
    v["i477"] = 0.076000 * np.tanh((((np.tanh((1.0))) < (np.where(data["CC_AMT_TOTAL_RECEIVABLE_MAX"] > 0, data["CC_AMT_TOTAL_RECEIVABLE_MAX"], ((((((data["CC_COUNT"]) > (
        data["CC_NAME_CONTRACT_STATUS_Active_SUM"])) * 1.)) < ((-1.0 * ((((data["CC_AMT_TOTAL_RECEIVABLE_MAX"]) * (data["CC_NAME_CONTRACT_STATUS_Active_SUM"]))))))) * 1.)))) * 1.))
    v["i478"] = 0.058999 * np.tanh((((np.where(data["APPROVED_RATE_DOWN_PAYMENT_MAX"] < -99998, data["APPROVED_RATE_DOWN_PAYMENT_MAX"], np.where(data["PREV_RATE_DOWN_PAYMENT_MAX"] > 0, data[
                                   "APPROVED_RATE_DOWN_PAYMENT_MAX"], data["PREV_RATE_DOWN_PAYMENT_MEAN"]))) > (np.where(data["PREV_RATE_DOWN_PAYMENT_MEAN"] > 0, 1.570796, np.tanh((data["APPROVED_RATE_DOWN_PAYMENT_MEAN"]))))) * 1.))
    v["i479"] = 0.098290 * np.tanh((((((data["REFUSED_APP_CREDIT_PERC_MAX"]) * ((((data["REFUSED_APP_CREDIT_PERC_MEAN"]) < ((((data["REFUSED_APP_CREDIT_PERC_MEAN"]) < (
        data["APPROVED_AMT_DOWN_PAYMENT_MEAN"])) * 1.))) * 1.)))) > ((((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) > (((data["APPROVED_RATE_DOWN_PAYMENT_MAX"]) / 2.0))) * 1.))) * 1.))
    v["i480"] = 0.100000 * np.tanh(((np.minimum(((((1.0) - (data["DAYS_BIRTH"])))), (((((((data["DAYS_BIRTH"]) * (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]))) < (
        (((data["DAYS_BIRTH"]) < (data["BURO_DAYS_CREDIT_VAR"])) * 1.))) * 1.))))) * (data["NAME_FAMILY_STATUS_Single___not_married"])))
    v["i481"] = 0.063000 * np.tanh(np.where(data["NAME_FAMILY_STATUS_Single___not_married"] < 0, ((np.where(data["BURO_DAYS_CREDIT_ENDDATE_MEAN"] > 0, np.minimum(((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"])),
                                                                                                                                                                  ((data["FLAG_EMP_PHONE"]))), (((data["NAME_FAMILY_STATUS_Single___not_married"]) > (data["FLAG_EMP_PHONE"])) * 1.))) / 2.0), ((data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]) * 2.0)))
    v["i482"] = 0.098001 * np.tanh((((((np.where(data["CLOSED_AMT_CREDIT_SUM_SUM"] < -99998, data["NAME_INCOME_TYPE_Student"], data["BURO_DAYS_CREDIT_ENDDATE_MEAN"])) > (
        ((3.0) - (np.where(data["BURO_DAYS_CREDIT_VAR"] < -99998, 3.0, data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]))))) * 1.)) * (data["BURO_DAYS_CREDIT_VAR"])))
    v["i483"] = 0.099230 * np.tanh((((7.0)) * ((((data["BURO_DAYS_CREDIT_VAR"]) > (np.where(data["BURO_DAYS_CREDIT_ENDDATE_MEAN"] < 0, np.where(data["NAME_FAMILY_STATUS_Single___not_married"]
                                                                                                                                                < 0, np.where(data["EXT_SOURCE_3"] < -99998, data["NAME_FAMILY_STATUS_Single___not_married"], (7.0)), data["NAME_FAMILY_STATUS_Single___not_married"]), (2.46100592613220215)))) * 1.))))
    v["i484"] = 0.099970 * np.tanh((((((1.570796) < (data["BURO_DAYS_CREDIT_VAR"])) * 1.)) * (np.where(np.where(data["AMT_INCOME_TOTAL"] > 0, data["EXT_SOURCE_3"], data[
                                   "BURO_DAYS_CREDIT_ENDDATE_MEAN"]) > 0, np.maximum(((data["EXT_SOURCE_3"])), ((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]))), (-1.0 * ((1.570796)))))))
    v["i485"] = 0.027810 * np.tanh((-1.0 * ((np.where(data["CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN"] < 0, np.where(data["BURO_DAYS_CREDIT_VAR"] < 0, ((((((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) < (
        data["BURO_DAYS_CREDIT_ENDDATE_MEAN"])) * 1.)) > ((((data["BURO_DAYS_CREDIT_VAR"]) > (-2.0)) * 1.))) * 1.), data["CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN"]), -2.0)))))
    v["i486"] = 0.099950 * np.tanh((((((3.0) - ((((((data["AMT_REQ_CREDIT_BUREAU_QRT"]) * 2.0)) + (((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]) + (((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]) * 2.0))))) / 2.0))))
                                     < (np.where(data["BURO_DAYS_CREDIT_ENDDATE_MEAN"] < 0, data["AMT_REQ_CREDIT_BUREAU_DAY"], data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]))) * 1.))
    v["i487"] = 0.099950 * np.tanh((((((((((((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) + (0.636620))) < (data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"])) * 1.))
                                        * 2.0)) * 2.0)) * (((((data["DAYS_BIRTH"]) + (data["DAYS_BIRTH"]))) + (data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"])))))
    v["i488"] = 0.069600 * np.tanh(((((((1.0)) < (((data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]) * 2.0))) * 1.)) * (np.where(data["EXT_SOURCE_3"] < 0, data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"],
                                                                                                                           np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"] > 0, -2.0, np.maximum(((data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"])), (((3.15189313888549805)))))))))
    v["i489"] = 0.098019 * np.tanh((-1.0 * (((((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) > ((-1.0 * ((np.where(data["EXT_SOURCE_3"] < -99998, data[
                                   "FLAG_EMP_PHONE"], (((data["NAME_FAMILY_STATUS_Single___not_married"]) + ((-1.0 * ((((data["EXT_SOURCE_3"]) / 2.0)))))) / 2.0))))))) * 1.)))))
    v["i490"] = 0.057200 * np.tanh((((((((((((((((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) < (1.570796)) * 1.)) + (data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"])) / 2.0)) > (
        0.318310)) * 1.)) + (0.0)) / 2.0)) > ((((data["BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"]) < (data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"])) * 1.))) * 1.))
    v["i491"] = 0.092970 * np.tanh((((np.where(((data["NEW_INC_PER_CHLD"]) * (((((((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]) > (data["NEW_INC_PER_CHLD"])) * 1.)) < (
        data["APPROVED_RATE_DOWN_PAYMENT_MEAN"])) * 1.))) > 0, (((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]) > (0.318310)) * 1.), (8.0))) < (data["APPROVED_AMT_DOWN_PAYMENT_MAX"])) * 1.))
    v["i492"] = 0.056000 * np.tanh((((((((((1.570796) < (data["BURO_DAYS_CREDIT_ENDDATE_MEAN"])) * 1.)) * (((((data["EXT_SOURCE_3"]) * 2.0)) * 2.0)))) + (
        ((((((data["EXT_SOURCE_3"]) < (1.570796)) * 1.)) < (data["BURO_DAYS_CREDIT_ENDDATE_MEAN"])) * 1.)))) * 2.0))
    v["i493"] = 0.098800 * np.tanh((-1.0 * ((np.minimum((((((data["FLOORSMAX_AVG"]) > ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])) * 1.))) * 1.))),
                                                        ((((((((data["FLOORSMAX_MODE"]) > ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (0.636620)) * 1.))) * 1.)) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.))))))))
    v["i494"] = 0.099420 * np.tanh(((((((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]) + (data["LIVINGAREA_MODE"])) / 2.0)) > (np.where(data["ELEVATORS_AVG"]
                                                                                                                                 > 0, np.where(data["REGION_POPULATION_RELATIVE"] > 0, ((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]) * (data["ELEVATORS_AVG"])), 3.0), 0.636620))) * 1.))
    v["i495"] = 0.099459 * np.tanh(((data["DAYS_BIRTH"]) * ((-1.0 * ((np.where(data["NEW_SOURCES_PROD"] < 0, ((data["DAYS_BIRTH"]) * ((-1.0 * (
        ((((data["DAYS_BIRTH"]) < (data["NEW_SOURCES_PROD"])) * 1.)))))), np.maximum(((data["NEW_SOURCES_PROD"])), ((data["DAYS_BIRTH"]))))))))))
    v["i496"] = 0.021039 * np.tanh(np.where(data["NEW_SOURCES_PROD"] < 0, data["OCCUPATION_TYPE_Secretaries"], (-1.0 * (((((((data["NEW_SOURCES_PROD"]) > (
        (((np.tanh((data["NEW_SOURCES_PROD"]))) > ((((data["NEW_SOURCES_PROD"]) + (((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]) * 2.0))) / 2.0))) * 1.))) * 1.)) * 2.0))))))
    v["i497"] = 0.099830 * np.tanh((((data["OCCUPATION_TYPE_Laborers"]) < (((((np.minimum(((data["NAME_INCOME_TYPE_Student"])), ((
        ((((data["NEW_SOURCES_PROD"]) * 2.0)) - (data["EXT_SOURCE_1"])))))) * 2.0)) - (((data["NEW_SOURCES_PROD"]) / 2.0))))) * 1.))
    v["i498"] = 0.038590 * np.tanh(((np.where(data["ORGANIZATION_TYPE_Military"] > 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], (-1.0 * (((((((np.maximum(((data["ORGANIZATION_TYPE_Military"])),
                                                                                                                                                     ((data["NEW_SOURCES_PROD"])))) + (np.maximum(((data["ORGANIZATION_TYPE_Military"])), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))) > (2.0)) * 1.)))))) * 2.0))
    v["i499"] = 0.046598 * np.tanh(((((((data["EXT_SOURCE_1"]) < (np.minimum((((((data["OCCUPATION_TYPE_Laborers"]) < (data["NEW_SOURCES_PROD"])) * 1.))), ((
        data["NEW_SOURCES_PROD"]))))) * 1.)) + ((-1.0 * (((((-2.0) < (np.minimum(((data["EXT_SOURCE_1"])), ((data["NEW_SOURCES_PROD"]))))) * 1.)))))) / 2.0))
    v["i500"] = 0.099900 * np.tanh((((((data["NEW_SOURCES_PROD"]) > (np.where(data["NEW_SOURCES_PROD"] < 0, data["NAME_INCOME_TYPE_Student"], np.maximum(
        (((3.0))), ((data["AMT_ANNUITY"])))))) * 1.)) * (np.where(data["AMT_ANNUITY"] > 0, (4.22247409820556641), data["NEW_SOURCES_PROD"]))))
    v["i501"] = 0.095040 * np.tanh((((((((3.0) < (np.maximum(((data["NEW_SOURCES_PROD"])), ((np.where(np.maximum(((data["AMT_ANNUITY"])), ((data["OCCUPATION_TYPE_Laborers"])))
                                                                                                      < 0, ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"])), data["AMT_INCOME_TOTAL"])))))) * 1.)) * 2.0)) * 2.0))
    v["i502"] = 0.001000 * np.tanh((((((data["APPROVED_RATE_DOWN_PAYMENT_MAX"]) > ((3.0))) * 1.)) * (((data["APPROVED_RATE_DOWN_PAYMENT_MAX"]) * (
        (((((data["APPROVED_RATE_DOWN_PAYMENT_MAX"]) > (3.141593)) * 1.)) + (((((data["APPROVED_RATE_DOWN_PAYMENT_MAX"]) * 2.0)) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))))))
    v["i503"] = 0.099960 * np.tanh(((np.where((((data["EXT_SOURCE_1"]) + ((((0.318310) + (data["APPROVED_AMT_DOWN_PAYMENT_MAX"])) / 2.0))) / 2.0) < 0, 0.0, (-1.0 * (
        ((((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]) > (np.maximum(((data["NEW_SOURCES_PROD"])), ((data["NAME_INCOME_TYPE_Student"]))))) * 1.)))))) * 2.0))
    v["i504"] = 0.000009 * np.tanh(((((data["APPROVED_RATE_DOWN_PAYMENT_MAX"]) - ((((data["APPROVED_RATE_DOWN_PAYMENT_MEAN"]) > (np.maximum(((data["APPROVED_AMT_DOWN_PAYMENT_MAX"])), ((
        data["APPROVED_RATE_DOWN_PAYMENT_MAX"]))))) * 1.)))) * ((((data["NEW_SOURCES_PROD"]) > ((((np.tanh((data["APPROVED_RATE_DOWN_PAYMENT_MEAN"]))) < (data["APPROVED_AMT_DOWN_PAYMENT_MAX"])) * 1.))) * 1.))))
    v["i505"] = 0.014603 * np.tanh((((((np.tanh((data["NEW_SOURCES_PROD"]))) > ((((data["PREV_AMT_DOWN_PAYMENT_MAX"]) > (np.tanh((np.tanh(
        (data["APPROVED_RATE_DOWN_PAYMENT_MEAN"])))))) * 1.))) * 1.)) * (((((data["APPROVED_RATE_DOWN_PAYMENT_MAX"]) + (-1.0))) + (-1.0)))))
    v["i506"] = 0.064800 * np.tanh(((((((data["NEW_SOURCES_PROD"]) < (((data["PREV_AMT_DOWN_PAYMENT_MEAN"]) * 2.0))) * 1.)) < (
        (((((((((((data["APPROVED_RATE_DOWN_PAYMENT_MEAN"]) < (((data["PREV_AMT_DOWN_PAYMENT_MEAN"]) / 2.0))) * 1.)) > (data["NEW_SOURCES_PROD"])) * 1.)) / 2.0)) + (data["PREV_AMT_DOWN_PAYMENT_MAX"])) / 2.0))) * 1.))
    v["i507"] = 0.099096 * np.tanh((((((np.where(data["NEW_SOURCES_PROD"] < -99998, data["AMT_ANNUITY"], data["NEW_SOURCES_PROD"])) < (data["EXT_SOURCE_1"])) * 1.)) * (
        np.where(data["NEW_SOURCES_PROD"] > 0, data["AMT_ANNUITY"], np.minimum(((data["EXT_SOURCE_1"])), (((-1.0 * ((data["NEW_SOURCES_PROD"]))))))))))
    v["i508"] = 0.052010 * np.tanh((-1.0 * (((((((np.where(data["EXT_SOURCE_1"] < -99998, 3.141593, ((data["EXT_SOURCE_1"]) * (((((data["EXT_SOURCE_1"]) - (
        data["APPROVED_AMT_DOWN_PAYMENT_MEAN"]))) - (data["NAME_INCOME_TYPE_Student"])))))) < (data["APPROVED_AMT_DOWN_PAYMENT_MEAN"])) * 1.)) * 2.0)))))
    v["i509"] = 0.099609 * np.tanh((-1.0 * (((((data["NEW_SOURCES_PROD"]) > (np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] < 0, ((
        (data["EXT_SOURCE_1"]) > (np.tanh((0.636620)))) * 1.), np.where(np.tanh((data["EXT_SOURCE_1"])) < 0, 0.318310, data["AMT_ANNUITY"])))) * 1.)))))
    v["i510"] = 0.041999 * np.tanh((((-1.0 * (((((np.maximum((((((2.0) + (data["BURO_DAYS_CREDIT_ENDDATE_MEAN"])) / 2.0))), ((
        ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["ORGANIZATION_TYPE_Emergency"])) / 2.0))))))) > (3.141593)) * 1.))))) * 2.0))
    v["i511"] = 0.092000 * np.tanh(((((data["OCCUPATION_TYPE_Laborers"]) * 2.0)) * (((((-1.0 * ((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])))) < (np.minimum(
        ((((((((data["OCCUPATION_TYPE_Laborers"]) > (data["AMT_ANNUITY"])) * 1.)) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) / 2.0))), ((data["AMT_ANNUITY"]))))) * 1.))))
    return Output(v.sum(axis=1))


def GP3(data):
    v = pd.DataFrame()
    v["i0"] = 0.059995 * np.tanh(((data["NAME_INCOME_TYPE_Working"]) + (((((((data["REGION_RATING_CLIENT_W_CITY"]) + (
        ((((data["DAYS_BIRTH"]) + ((((data["REFUSED_AMT_CREDIT_MEAN"]) > (-2.0)) * 1.)))) * 2.0)))) + (np.tanh((data["REFUSED_AMT_CREDIT_MAX"]))))) * 2.0))))
    v["i1"] = 0.099760 * np.tanh(((((((data["REGION_RATING_CLIENT_W_CITY"]) + ((((data["DAYS_BIRTH"]) + (data["NAME_INCOME_TYPE_Working"])) / 2.0)))) + (
        np.where(data["NAME_EDUCATION_TYPE_Higher_education"] > 0, -2.0, np.tanh((((data["DAYS_BIRTH"]) - (data["CLOSED_AMT_CREDIT_SUM_MEAN"])))))))) * 2.0))
    v["i2"] = 0.099250 * np.tanh((((7.0)) * (((np.maximum(((data["REFUSED_CNT_PAYMENT_SUM"])), ((((((((((data["DAYS_BIRTH"]) + (
        0.636620))) + (data["DAYS_BIRTH"])) / 2.0)) + (np.tanh((((data["REFUSED_DAYS_DECISION_MAX"]) * 2.0))))) / 2.0))))) * 2.0))))
    v["i3"] = 0.099904 * np.tanh(((((np.where(data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"] < 0, (-1.0 * ((np.where(data["NAME_EDUCATION_TYPE_Higher_education"]
                                                                                                             > 0, (9.39576244354248047), ((data["CLOSED_AMT_CREDIT_SUM_SUM"]) * ((14.40538311004638672))))))), (14.40538311004638672))) - (3.0))) * 2.0))
    v["i4"] = 0.099895 * np.tanh(((((((((((((data["REGION_RATING_CLIENT_W_CITY"]) + (((np.maximum((((((data["REFUSED_DAYS_DECISION_MAX"]) + (
        2.0)) / 2.0))), ((data["REGION_RATING_CLIENT_W_CITY"])))) - (data["CODE_GENDER"]))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0))
    v["i5"] = 0.099954 * np.tanh(((((np.where(data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"] < -99998, 1.570796, data["NEW_SCORES_STD"])) + (((np.where(
        data["NEW_DOC_IND_KURT"] > 0, data["DAYS_BIRTH"], data["NEW_DOC_IND_KURT"])) + ((((data["DAYS_ID_PUBLISH"]) + (data["REG_CITY_NOT_WORK_CITY"])) / 2.0)))))) * 2.0))
    v["i6"] = 0.099953 * np.tanh(((np.where(np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"] > 0, (3.0), np.minimum(((((data["NEW_SCORES_STD"]) - (data["NAME_EDUCATION_TYPE_Higher_education"])))),
                                                                                                                     ((data["NEW_SCORES_STD"])))) < 0, ((data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"]) + (data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"])), (3.97629356384277344))) * 2.0))
    v["i7"] = 0.099980 * np.tanh(np.where(data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"] < 0, ((((((np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, data[
                                 "REG_CITY_NOT_LIVE_CITY"], 1.0)) + (((data["FLAG_DOCUMENT_3"]) + (data["REGION_RATING_CLIENT_W_CITY"]))))) + (data["DAYS_ID_PUBLISH"]))) * 2.0), 3.141593))
    v["i8"] = 0.099978 * np.tanh(((np.where(data["EXT_SOURCE_1"] < -99998, ((np.where(data["CODE_GENDER"] < 0, 3.0, data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"])
                                                                             ) - (-1.0)), ((((np.tanh((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]))) - (data["EXT_SOURCE_1"]))) * 2.0))) * 2.0))
    v["i9"] = 0.099850 * np.tanh((-1.0 * ((np.where(data["NEW_CAR_TO_BIRTH_RATIO"] < -99998, ((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) + (np.where(
        data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"] < 0, 3.141593, -2.0))), ((np.where(data["NEW_CAR_TO_EMPLOY_RATIO"] < 0, data["NEW_CAR_TO_BIRTH_RATIO"], 3.141593)) * 2.0))))))
    v["i10"] = 0.099852 * np.tanh(((1.0) + (((((((data["REFUSED_AMT_CREDIT_MAX"]) + (((np.tanh((data["REFUSED_DAYS_DECISION_MEAN"]))) + (
        ((((data["REFUSED_AMT_CREDIT_MAX"]) - (data["REFUSED_AMT_APPLICATION_MEAN"]))) * 2.0)))))) - (data["REFUSED_AMT_APPLICATION_MEAN"]))) * 2.0))))
    v["i11"] = 0.099782 * np.tanh((((((-1.0 * ((data["FLOORSMAX_MODE"])))) - (np.where(data["FLOORSMAX_AVG"] < -99998, (-1.0 * (
        (data["ACTIVE_DAYS_CREDIT_VAR"]))), np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, 1.0, data["ACTIVE_DAYS_CREDIT_VAR"]))))) * 2.0))
    v["i12"] = 0.099950 * np.tanh(((((((((((np.maximum(((np.maximum(((((data["CC_CNT_DRAWINGS_POS_CURRENT_MEAN"]) * (data["NAME_FAMILY_STATUS_Married"])))), ((data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"]))))),
                                                       ((data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"])))) + (data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]))) * 2.0)) * 2.0)) + (data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]))) * 2.0))
    v["i13"] = 0.099900 * np.tanh(((((((((np.where(data["NEW_SOURCES_PROD"] < -99998, data["NEW_DOC_IND_KURT"], ((
        (-1.0 * ((((1.570796) + (((data["NEW_SOURCES_PROD"]) * 2.0))))))) * 2.0))) * 2.0)) * 2.0)) * 2.0)) * 2.0))
    v["i14"] = 0.099890 * np.tanh(((np.where(data["OCCUPATION_TYPE_Core_staff"] < 0, ((((data["APPROVED_RATE_DOWN_PAYMENT_MEAN"]) - (data["CODE_GENDER"]))) - (
        (((2.0)) + ((((9.94526481628417969)) * (data["APPROVED_AMT_DOWN_PAYMENT_MAX"])))))), -2.0)) - (data["NAME_INCOME_TYPE_State_servant"])))
    v["i15"] = 0.099653 * np.tanh(((((((data["REG_CITY_NOT_LIVE_CITY"]) + ((-1.0 * ((np.where(data["NEW_CAR_TO_BIRTH_RATIO"] > 0, 3.0, (-1.0 * (
        (data["REGION_RATING_CLIENT"])))))))))) + (((data["DAYS_REGISTRATION"]) - (data["FLAG_OWN_CAR"]))))) + (data["DAYS_ID_PUBLISH"])))
    v["i16"] = 0.099990 * np.tanh(np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"] < 0, np.where(data["CODE_GENDER"] < 0, np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"] > 0,
                                                                                                                                    (7.63108682632446289), 0.318310), ((((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) * 2.0)) * 2.0)) + (data["CC_CNT_DRAWINGS_POS_CURRENT_MEAN"]))), (7.63108682632446289)))
    v["i17"] = 0.099948 * np.tanh(((((np.maximum(((data["NEW_SCORES_STD"])), ((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"])))) + ((((data["NEW_CREDIT_TO_INCOME_RATIO"]) + (data["NAME_INCOME_TYPE_Working"])) / 2.0)))
                                    ) + (np.where(data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"] < 0, np.minimum(((data["ORGANIZATION_TYPE_Self_employed"])), ((data["NEW_CREDIT_TO_INCOME_RATIO"]))), 1.570796))))
    v["i18"] = 0.099961 * np.tanh(((((data["ACTIVE_DAYS_CREDIT_VAR"]) - (np.where(((data["APPROVED_RATE_DOWN_PAYMENT_MAX"]) + (data["APPROVED_AMT_DOWN_PAYMENT_MAX"]))
                                                                                  < 0, data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"], (7.0))))) - (((((((np.tanh((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]))) * 2.0)) * 2.0)) * 2.0))))
    v["i19"] = 0.099571 * np.tanh(((((((((data["ORGANIZATION_TYPE_Business_Entity_Type_3"]) + (((data["ORGANIZATION_TYPE_Transport__type_3"]) + (data["ORGANIZATION_TYPE_Construction"]))))) + (
        ((data["NAME_INCOME_TYPE_Unemployed"]) * 2.0)))) - (data["FLAG_PHONE"]))) + (((data["FLAG_WORK_PHONE"]) + (data["ORGANIZATION_TYPE_Self_employed"])))))
    v["i20"] = 0.099995 * np.tanh(np.where((((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) > (data["OCCUPATION_TYPE_High_skill_tech_staff"])) * 1.) > 0, (13.08571720123291016), np.where(
        data["NEW_CAR_TO_BIRTH_RATIO"] > 0, -2.0, ((((data["NEW_DOC_IND_KURT"]) - (np.maximum(((data["FLAG_OWN_CAR"])), ((data["OCCUPATION_TYPE_High_skill_tech_staff"])))))) / 2.0))))
    v["i21"] = 0.098980 * np.tanh((-1.0 * ((((np.where(((((0.42689332365989685)) + (data["CC_CNT_INSTALMENT_MATURE_CUM_VAR"])) / 2.0) < 0, np.maximum(
        ((((data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"]) - (data["WALLSMATERIAL_MODE_Panel"])))), ((data["WALLSMATERIAL_MODE_Panel"]))), data["CC_MONTHS_BALANCE_VAR"])) + (data["OCCUPATION_TYPE_Accountants"]))))))
    v["i22"] = 0.098079 * np.tanh(np.where(data["ORGANIZATION_TYPE_Military"] > 0, -2.0, np.maximum(((((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) + (
        (((data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"]) < (data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"])) * 1.)))) + (np.tanh((data["CC_CNT_DRAWINGS_POS_CURRENT_MEAN"])))))), (((-1.0 * ((0.318310))))))))
    v["i23"] = 0.098040 * np.tanh(((((data["ORGANIZATION_TYPE_Transport__type_3"]) + (data["NAME_INCOME_TYPE_Unemployed"]))) - ((((((0.636620) + (np.tanh(
        (((data["CC_NAME_CONTRACT_STATUS_Approved_VAR"]) + (((data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"]) + (data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"])))))))) / 2.0)) * 2.0))))
    v["i24"] = 0.099955 * np.tanh(np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"] < 0, np.maximum(((data["CC_CNT_DRAWINGS_OTHER_CURRENT_MAX"])), ((((np.tanh((1.0))) + (
        np.tanh((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]))))))), ((((data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"]) + (data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]))) * 2.0)))
    v["i25"] = 0.099954 * np.tanh(((((((-1.0 * ((0.318310)))) < (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])) * 1.)) + (np.minimum(
        (((-1.0 * ((((((data["EXT_SOURCE_1"]) * 2.0)) * 2.0)))))), (((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) > ((-1.0 * ((0.318310))))) * 1.)))))))
    v["i26"] = 0.099800 * np.tanh((((np.maximum(((data["OCCUPATION_TYPE_Low_skill_Laborers"])), ((np.maximum(((data["ORGANIZATION_TYPE_Trade__type_3"])), ((data["NEW_SCORES_STD"]))))))) + (
        np.minimum(((((((1.570796) + (((((data["NEW_CREDIT_TO_INCOME_RATIO"]) * 2.0)) * 2.0)))) * 2.0))), ((data["ORGANIZATION_TYPE_Trade__type_3"]))))) / 2.0))
    v["i27"] = 0.099530 * np.tanh((((((-1.0 * (((((((data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]) > (data["ORGANIZATION_TYPE_Realtor"])) * 1.)) - (
        ((3.0) * ((((data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]) > ((((1.35391867160797119)) * (data["NAME_INCOME_TYPE_Unemployed"])))) * 1.))))))))) * 2.0)) * 2.0))
    v["i28"] = 0.099897 * np.tanh((((((((((np.maximum(((data["OCCUPATION_TYPE_Drivers"])), ((data["ORGANIZATION_TYPE_Industry__type_9"])))) - (data["NAME_EDUCATION_TYPE_Incomplete_higher"]))) + (
        ((((-1.0 * ((data["HOUR_APPR_PROCESS_START"])))) + (data["NAME_HOUSING_TYPE_Municipal_apartment"])) / 2.0))) / 2.0)) - (data["ORGANIZATION_TYPE_Military"]))) - (data["ORGANIZATION_TYPE_Industry__type_9"])))
    v["i29"] = 0.099930 * np.tanh((-1.0 * ((((((((data["AMT_CREDIT"]) * (((data["AMT_CREDIT"]) * 2.0)))) - (
        (((0.318310) > (((data["AMT_CREDIT"]) * 2.0))) * 1.)))) - ((((0.636620) + (0.318310)) / 2.0)))))))
    v["i30"] = 0.099996 * np.tanh((((((data["ORGANIZATION_TYPE_Realtor"]) - (np.tanh((((data["APPROVED_RATE_DOWN_PAYMENT_MAX"]) / 2.0)))))) + (((data["ORGANIZATION_TYPE_Restaurant"]) - (
        np.maximum(((((data["ORGANIZATION_TYPE_Police"]) + (data["ORGANIZATION_TYPE_Security_Ministries"])))), ((data["APPROVED_RATE_DOWN_PAYMENT_MAX"]))))))) / 2.0))
    v["i31"] = 0.099794 * np.tanh(((data["INSTAL_AMT_INSTALMENT_MAX"]) + (((3.141593) * ((((((8.91020011901855469)) * (
        ((((np.where(data["NAME_INCOME_TYPE_Unemployed"] > 0, 3.0, data["INSTAL_DPD_MEAN"])) * 2.0)) * 2.0)))) + (3.141593)))))))
    v["i32"] = 0.099959 * np.tanh(np.minimum(((np.where(data["CC_AMT_DRAWINGS_CURRENT_VAR"] < -99998, 0.318310, ((data["CC_CNT_INSTALMENT_MATURE_CUM_VAR"]) + ((((data["CC_AMT_DRAWINGS_CURRENT_VAR"]) > (
        ((data["CC_CNT_INSTALMENT_MATURE_CUM_VAR"]) / 2.0))) * 1.)))))), (((-1.0 * ((((data["CC_CNT_INSTALMENT_MATURE_CUM_VAR"]) + (data["CC_NAME_CONTRACT_STATUS_Signed_VAR"])))))))))
    v["i33"] = 0.099696 * np.tanh(np.where((((((((data["ACTIVE_AMT_ANNUITY_MEAN"]) < (((data["ACTIVE_AMT_ANNUITY_MAX"]) * (1.570796)))) * 1.)) + (data["ORGANIZATION_TYPE_Realtor"]))) + (
        data["NAME_INCOME_TYPE_Unemployed"])) < 0, ((data["NAME_EDUCATION_TYPE_Lower_secondary"]) + (data["ORGANIZATION_TYPE_Transport__type_3"])), 3.0))
    v["i34"] = 0.099980 * np.tanh(((((data["ORGANIZATION_TYPE_Trade__type_7"]) - (np.where((((0.636620) + (data["CLOSED_DAYS_CREDIT_VAR"])) / 2.0) < 0, ((
        data["ORGANIZATION_TYPE_Medicine"]) * 2.0), np.maximum((((0.48020493984222412))), ((data["ORGANIZATION_TYPE_School"]))))))) - (data["OCCUPATION_TYPE_Medicine_staff"])))
    v["i35"] = 0.099982 * np.tanh(np.where(data["EXT_SOURCE_3"] < -99998, ((data["NAME_INCOME_TYPE_Maternity_leave"]
                                                                            ) - (0.636620)), ((-2.0) - ((((8.16910552978515625)) * (data["EXT_SOURCE_3"]))))))
    v["i36"] = 0.097950 * np.tanh(np.tanh(((((((data["NEW_CREDIT_TO_INCOME_RATIO"]) / 2.0)) + ((-1.0 * ((((data["AMT_REQ_CREDIT_BUREAU_YEAR"]) + (
        np.where((((-1.0) < (data["NEW_CREDIT_TO_INCOME_RATIO"])) * 1.) > 0, data["WEEKDAY_APPR_PROCESS_START_SUNDAY"], (8.82685852050781250))))))))) / 2.0))))
    v["i37"] = 0.099970 * np.tanh(np.maximum((((((-1.0) + (data["NAME_INCOME_TYPE_Unemployed"])) / 2.0))), ((((((data["CLOSED_DAYS_CREDIT_MEAN"]) + (
        ((((data["CLOSED_AMT_CREDIT_SUM_DEBT_MEAN"]) * 2.0)) + (data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"]))))) - (np.tanh((np.tanh((data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]))))))))))
    v["i38"] = 0.099897 * np.tanh((((((((((((((data["INSTAL_AMT_INSTALMENT_MAX"]) - (data["ORGANIZATION_TYPE_Bank"]))) + (
        (((((-1.0) > (((data["INSTAL_AMT_INSTALMENT_MAX"]) * 2.0))) * 1.)) * 2.0)))) * 2.0)) * 2.0)) + (-1.0)) / 2.0)) * 2.0))
    v["i39"] = 0.099808 * np.tanh(((((((np.tanh((data["APPROVED_RATE_DOWN_PAYMENT_MEAN"]))) - (np.minimum(((data["APPROVED_RATE_DOWN_PAYMENT_MAX"])), ((data["APPROVED_AMT_DOWN_PAYMENT_MAX"])))))
                                      ) + (np.tanh((((data["APPROVED_RATE_DOWN_PAYMENT_MEAN"]) - (data["APPROVED_RATE_DOWN_PAYMENT_MAX"]))))))) + (data["AMT_INCOME_TOTAL"])))
    v["i40"] = 0.099997 * np.tanh(((((((((data["ORGANIZATION_TYPE_XNA"]) - (data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]))) - (data["NEW_EMPLOY_TO_BIRTH_RATIO"]))) * ((4.0)))) - (np.where(
        data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"] > 0, (((data["NEW_EMPLOY_TO_BIRTH_RATIO"]) < (data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"])) * 1.), data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]))))
    v["i41"] = 0.099999 * np.tanh((((data["PREV_APP_CREDIT_PERC_MEAN"]) + (((np.where(data["PREV_APP_CREDIT_PERC_MEAN"] > 0, ((-1.0) - (data["PREV_AMT_GOODS_PRICE_MEAN"])), ((
        (data["PREV_APP_CREDIT_PERC_MEAN"]) < (np.maximum(((data["OCCUPATION_TYPE_Laborers"])), ((data["PREV_APP_CREDIT_PERC_MAX"]))))) * 1.))) * 2.0))) / 2.0))
    v["i42"] = 0.099952 * np.tanh((((((((data["ORGANIZATION_TYPE_Construction"]) - (((data["NAME_FAMILY_STATUS_Widow"]) - (((data["WALLSMATERIAL_MODE_Panel"]) - (
        data["NAME_INCOME_TYPE_Pensioner"]))))))) / 2.0)) + (((((data["NAME_INCOME_TYPE_Pensioner"]) * (((data["NAME_FAMILY_STATUS_Widow"]) / 2.0)))) / 2.0))) / 2.0))
    v["i43"] = 0.098404 * np.tanh((((data["ORGANIZATION_TYPE_Realtor"]) + ((((np.where(data["AMT_INCOME_TOTAL"] < 0, ((data["AMT_INCOME_TOTAL"]) + (data["OCCUPATION_TYPE_Cooking_staff"])), data[
                                  "ORGANIZATION_TYPE_Realtor"])) + ((-1.0 * ((np.maximum(((data["WEEKDAY_APPR_PROCESS_START_MONDAY"])), ((data["NAME_HOUSING_TYPE_Office_apartment"])))))))) / 2.0))) / 2.0))
    v["i44"] = 0.096048 * np.tanh(((((-1.0 * (((((np.tanh((data["NEW_CREDIT_TO_INCOME_RATIO"]))) + (data["WEEKDAY_APPR_PROCESS_START_SATURDAY"])) / 2.0))))) + ((((data["NEW_CREDIT_TO_INCOME_RATIO"]) + (
        ((((np.maximum(((data["ORGANIZATION_TYPE_Transport__type_4"])), ((data["ORGANIZATION_TYPE_Industry__type_3"])))) - (data["REG_REGION_NOT_LIVE_REGION"]))) * 2.0))) / 2.0))) / 2.0))
    v["i45"] = 0.099740 * np.tanh(((np.where(data["REFUSED_AMT_DOWN_PAYMENT_MAX"] < -99998, data["NAME_HOUSING_TYPE_Rented_apartment"], (((data["NAME_HOUSING_TYPE_Rented_apartment"]) < (data["AMT_INCOME_TOTAL"])) * 1.))) - (
        np.where((((data["NAME_HOUSING_TYPE_Rented_apartment"]) < (data["AMT_INCOME_TOTAL"])) * 1.) > 0, ((data["FLAG_OWN_REALTY"]) / 2.0), data["ORGANIZATION_TYPE_Security_Ministries"]))))
    v["i46"] = 0.087100 * np.tanh((((((data["NAME_HOUSING_TYPE_Municipal_apartment"]) - (data["NAME_EDUCATION_TYPE_Academic_degree"]))) + ((((np.maximum(((data["NAME_INCOME_TYPE_Maternity_leave"])), ((
        data["ORGANIZATION_TYPE_Legal_Services"])))) + ((((((data["NAME_INCOME_TYPE_Unemployed"]) + (data["HOUSETYPE_MODE_specific_housing"]))) + ((-1.0 * ((data["ORGANIZATION_TYPE_Trade__type_2"]))))) / 2.0))) / 2.0))) / 2.0))
    v["i47"] = 0.099954 * np.tanh((((((((-1.0 * ((np.where(data["EXT_SOURCE_3"] < -99998, (-1.0 * ((((0.318310) / 2.0)))), data[
                                  "EXT_SOURCE_3"]))))) - (0.318310))) - ((((data["EXT_SOURCE_3"]) > (data["ORGANIZATION_TYPE_Realtor"])) * 1.)))) * 2.0))
    v["i48"] = 0.099929 * np.tanh(((np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"] < -99998, 0.318310, (((((((((data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]) > (
        (-1.0 * ((0.318310))))) * 1.)) - ((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) < (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])) * 1.)))) * 2.0)) * 2.0))) * 2.0))
    v["i49"] = 0.099700 * np.tanh(np.where((((data["CLOSED_CNT_CREDIT_PROLONG_SUM"]) < ((-1.0 * ((data["CLOSED_AMT_CREDIT_SUM_MEAN"]))))) * 1.) > 0, (((data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"]) > (
        (((((0.636620) * (data["CLOSED_AMT_CREDIT_SUM_MEAN"]))) + (data["CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN"])) / 2.0))) * 1.), (-1.0 * ((data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"])))))
    v["i50"] = 0.099990 * np.tanh((-1.0 * ((np.where(data["EXT_SOURCE_3"] < -99998, 0.636620, ((
        ((((((((-1.0 * ((data["EXT_SOURCE_3"])))) > (np.tanh((np.tanh((0.318310)))))) * 1.)) + (data["EXT_SOURCE_3"]))) * 2.0)) * 2.0))))))
    v["i51"] = 0.099807 * np.tanh(np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, ((((((0.636620) < (data["NEW_SCORES_STD"])) * 1.)) > (np.tanh((data["NEW_SCORES_STD"]))))
                                                                                     * 1.), ((data["NEW_CREDIT_TO_INCOME_RATIO"]) - (((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) - ((-1.0 * ((data["NEW_SCORES_STD"])))))))))
    v["i52"] = 0.097999 * np.tanh(np.tanh((((np.maximum((((((data["CLOSED_AMT_CREDIT_SUM_DEBT_MEAN"]) > (data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"])) * 1.))), ((
        ((((((data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"]) + ((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) + ((8.0))) / 2.0))) / 2.0)) > (3.0)) * 1.))))) - (0.318310)))))
    v["i53"] = 0.097000 * np.tanh(((data["NAME_INCOME_TYPE_Unemployed"]) - (np.where(((data["AMT_REQ_CREDIT_BUREAU_QRT"]) - (2.0)) > 0, data[
                                  "NAME_INCOME_TYPE_Maternity_leave"], ((np.where(data["AMT_REQ_CREDIT_BUREAU_QRT"] < 0, np.tanh((np.tanh((data["AMT_REQ_CREDIT_BUREAU_MON"])))), (10.0))) / 2.0)))))
    v["i54"] = 0.099000 * np.tanh((-1.0 * ((((((((data["REFUSED_DAYS_DECISION_MEAN"]) > ((((data["REFUSED_DAYS_DECISION_MAX"]) > (np.tanh((data["REFUSED_DAYS_DECISION_MEAN"])))) * 1.)))
                                                * 1.)) > (np.maximum(((np.tanh((data["REFUSED_DAYS_DECISION_MEAN"])))), (((-1.0 * ((data["REFUSED_HOUR_APPR_PROCESS_START_MEAN"])))))))) * 1.)))))
    v["i55"] = 0.099001 * np.tanh((((((((((data["NAME_TYPE_SUITE_Other_B"]) - ((((data["NAME_EDUCATION_TYPE_Academic_degree"]) + (data["OCCUPATION_TYPE_Core_staff"])) / 2.0)))) + (np.maximum(
        ((data["NAME_INCOME_TYPE_Unemployed"])), ((data["OCCUPATION_TYPE_Security_staff"]))))) / 2.0)) - ((((data["ORGANIZATION_TYPE_School"]) + (data["NAME_EDUCATION_TYPE_Academic_degree"])) / 2.0)))) / 2.0))
    v["i56"] = 0.054646 * np.tanh(((((data["NEW_INC_BY_ORG"]) * (data["NAME_TYPE_SUITE_Spouse__partner"]))) - (np.maximum(((np.maximum((((((np.maximum(((data["ORGANIZATION_TYPE_Industry__type_12"])), ((
        data["NAME_EDUCATION_TYPE_Academic_degree"])))) + (data["NAME_TYPE_SUITE_Spouse__partner"])) / 2.0))), ((((data["ORGANIZATION_TYPE_Industry__type_9"]) * 2.0)))))), ((data["ORGANIZATION_TYPE_Industry__type_9"]))))))
    v["i57"] = 0.099809 * np.tanh(((((((data["ORGANIZATION_TYPE_Transport__type_3"]) - (np.where(((data["NEW_SOURCES_PROD"]) + (
        data["ORGANIZATION_TYPE_Legal_Services"])) < -99998, data["ORGANIZATION_TYPE_Legal_Services"], ((data["NEW_SOURCES_PROD"]) + (0.636620)))))) * 2.0)) * 2.0))
    v["i58"] = 0.077390 * np.tanh(((((np.maximum(((data["ORGANIZATION_TYPE_Realtor"])), ((((np.tanh((np.tanh((data["NAME_FAMILY_STATUS_Separated"]))))) + (
        ((((data["OCCUPATION_TYPE_Cleaning_staff"]) * (data["OCCUPATION_TYPE_Cleaning_staff"]))) / 2.0))))))) - (data["NAME_EDUCATION_TYPE_Academic_degree"]))) - (data["ORGANIZATION_TYPE_Hotel"])))
    v["i59"] = 0.099879 * np.tanh(np.where(data["NEW_SOURCES_PROD"] < -99998, data["OCCUPATION_TYPE_Low_skill_Laborers"], ((
        (((((((((-1.0) > (data["NEW_SOURCES_PROD"])) * 1.)) - ((((np.tanh((0.318310))) < (data["NEW_SOURCES_PROD"])) * 1.)))) * 2.0)) * 2.0)) * 2.0)))
    v["i60"] = 0.096971 * np.tanh(np.where(data["NEW_SCORES_STD"] > 0, ((data["AMT_INCOME_TOTAL"]) + ((-1.0 * (
        (((data["NEW_SCORES_STD"]) * (((((data["FLAG_EMP_PHONE"]) / 2.0)) - (((data["NEW_SCORES_STD"]) + (-2.0))))))))))), 0.318310))
    v["i61"] = 0.099985 * np.tanh(((np.where(data["NEW_SOURCES_PROD"] < -99998, (8.0), data["NEW_SOURCES_PROD"])) * (np.minimum(((np.where(
        data["NEW_SOURCES_PROD"] < -99998, (8.0), data["NEW_SOURCES_PROD"]))), ((np.where(data["NEW_SOURCES_PROD"] < -99998, data["NAME_INCOME_TYPE_Unemployed"], -1.0)))))))
    v["i62"] = 0.099890 * np.tanh(((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) * (np.where(data["ACTIVE_MONTHS_BALANCE_MIN_MIN"] > 0, ((data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"]) * (
        data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"])), np.where(data["ACTIVE_MONTHS_BALANCE_MAX_MAX"] > 0, 1.570796, (-1.0 * (((((1.570796) < (data["NEW_CREDIT_TO_INCOME_RATIO"])) * 1.)))))))))
    v["i63"] = 0.099809 * np.tanh((-1.0 * ((((((((((np.maximum(((data["DAYS_BIRTH"])), (((((data["DAYS_BIRTH"]) < (np.maximum(((data["WEEKDAY_APPR_PROCESS_START_SUNDAY"])),
                                                                                                                              ((data["ORGANIZATION_TYPE_Security_Ministries"]))))) * 1.))))) - (1.0))) - (data["ORGANIZATION_TYPE_Realtor"]))) * 2.0)) * 2.0)))))
    v["i64"] = 0.095100 * np.tanh(np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, np.maximum(((data["NAME_CONTRACT_TYPE_Revolving_loans"])), ((np.maximum(((data["NAME_INCOME_TYPE_Unemployed"])),
                                                                                                                                                            ((data["OCCUPATION_TYPE_Waiters_barmen_staff"])))))), np.minimum(((((data["NEW_INC_BY_ORG"]) * (data["ACTIVE_DAYS_CREDIT_VAR"])))), (((-1.0 * ((data["NEW_INC_BY_ORG"]))))))))
    v["i65"] = 0.099989 * np.tanh(np.where(data["EXT_SOURCE_3"] < -99998, (((np.minimum(((data["DAYS_ID_PUBLISH"])), ((np.tanh((np.where(data["DAYS_ID_PUBLISH"] < 0, data[
                                  "ORGANIZATION_TYPE_Legal_Services"], -1.0))))))) + (data["DAYS_ID_PUBLISH"])) / 2.0), ((data["ORGANIZATION_TYPE_Legal_Services"]) - (data["EXT_SOURCE_3"]))))
    v["i66"] = 0.098501 * np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"] < -99998, 0.318310, (((((np.minimum(((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) - (
        data["ACTIVE_CNT_CREDIT_PROLONG_SUM"])))), ((data["ACTIVE_AMT_CREDIT_SUM_SUM"])))) - (np.tanh((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]))))) < (data["ACTIVE_CNT_CREDIT_PROLONG_SUM"])) * 1.)))
    v["i67"] = 0.099050 * np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_SUM"] > 0, ((((((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]) > (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])) * 1.)) > (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]))
                                                                                   * 1.), (-1.0 * ((np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, (((data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"]) < (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])) * 1.), (9.16476345062255859)))))))
    v["i68"] = 0.099690 * np.tanh((-1.0 * ((np.where(data["DAYS_EMPLOYED"] < -99998, ((data["DAYS_BIRTH"]) - (np.tanh(
        (((data["DAYS_BIRTH"]) * 2.0))))), np.maximum(((data["NEW_EMPLOY_TO_BIRTH_RATIO"])), ((((((data["DAYS_BIRTH"]) - (1.0))) * 2.0)))))))))
    v["i69"] = 0.099402 * np.tanh(np.maximum(((np.where(data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"] < -99998, data["CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN"], np.where(data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"] > 0, data[
                                  "CLOSED_CREDIT_DAY_OVERDUE_MEAN"], data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"])))), (((((data["NAME_INCOME_TYPE_Unemployed"]) + (((np.tanh((((data["CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN"]) / 2.0)))) / 2.0))) / 2.0)))))
    v["i70"] = 0.099710 * np.tanh(((np.maximum(((0.0)), ((np.where(data["NAME_FAMILY_STATUS_Married"] < 0, data["NEW_CREDIT_TO_INCOME_RATIO"], data["NAME_INCOME_TYPE_Maternity_leave"]))))) - (
        (((data["REFUSED_DAYS_DECISION_MEAN"]) > ((((data["REFUSED_DAYS_DECISION_MEAN"]) < (((((data["REFUSED_DAYS_DECISION_MAX"]) / 2.0)) * 2.0))) * 1.))) * 1.))))
    v["i71"] = 0.099490 * np.tanh(np.where(data["OCCUPATION_TYPE_Managers"] < 0, np.minimum((((((data["ORGANIZATION_TYPE_Legal_Services"]) > (data["AMT_CREDIT"])) * 1.))), ((
        ((data["AMT_CREDIT"]) + ((((np.maximum(((data["OCCUPATION_TYPE_Laborers"])), ((data["OCCUPATION_TYPE_Sales_staff"])))) > (data["AMT_CREDIT"])) * 1.)))))), data["OCCUPATION_TYPE_Laborers"]))
    v["i72"] = 0.099850 * np.tanh(np.where(((data["POS_NAME_CONTRACT_STATUS_Signed_MEAN"]) - (data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"])) < 0, data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"], ((
        data["POS_NAME_CONTRACT_STATUS_Signed_MEAN"]) - (np.where(data["DAYS_ID_PUBLISH"] < 0, np.tanh(((((data["POS_NAME_CONTRACT_STATUS_Signed_MEAN"]) > (data["NAME_INCOME_TYPE_Unemployed"])) * 1.))), 1.570796)))))
    v["i73"] = 0.099762 * np.tanh(np.where(np.tanh((data["DAYS_ID_PUBLISH"])) > 0, np.maximum((((((data["ORGANIZATION_TYPE_Legal_Services"]) < (
        (((-1.0) + (np.tanh((data["DAYS_ID_PUBLISH"])))) / 2.0))) * 1.))), ((data["ORGANIZATION_TYPE_Legal_Services"]))), ((data["DAYS_ID_PUBLISH"]) - (-1.0))))
    v["i74"] = 0.097890 * np.tanh((((((((data["NEW_SCORES_STD"]) > (3.141593)) * 1.)) + (np.where(data["NEW_SCORES_STD"] > 0, ((
        (-1.0 * ((3.141593)))) * (data["FLAG_CONT_MOBILE"])), ((data["NAME_HOUSING_TYPE_With_parents"]) * (data["FLAG_CONT_MOBILE"])))))) * 2.0))
    v["i75"] = 0.099601 * np.tanh((((((((data["NAME_INCOME_TYPE_Unemployed"]) - (data["ORGANIZATION_TYPE_Industry__type_12"]))) - (data["NAME_EDUCATION_TYPE_Academic_degree"]))) + (np.where(data["FONDKAPREMONT_MODE_reg_oper_account"]
                                                                                                                                                                                              > 0, (((data["WALLSMATERIAL_MODE_Stone__brick"]) > (data["NAME_EDUCATION_TYPE_Academic_degree"])) * 1.), ((data["ORGANIZATION_TYPE_Legal_Services"]) + (data["ORGANIZATION_TYPE_Legal_Services"]))))) / 2.0))
    v["i76"] = 0.097098 * np.tanh((-1.0 * ((((np.where(data["EXT_SOURCE_3"] < -99998, np.tanh((np.tanh((np.tanh(
        (data["EXT_SOURCE_3"])))))), data["EXT_SOURCE_3"])) - (np.tanh((np.tanh((((((data["EXT_SOURCE_3"]) * 2.0)) * 2.0)))))))))))
    v["i77"] = 0.099652 * np.tanh(np.where(data["ACTIVE_CREDIT_DAY_OVERDUE_MEAN"] > 0, (3.80976772308349609), ((-2.0) * (np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"])), ((
        (((np.minimum(((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"])), ((data["ACTIVE_CREDIT_DAY_OVERDUE_MEAN"])))) < (((data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]) * 2.0))) * 1.))))))))
    v["i78"] = 0.095980 * np.tanh((((-1.0 * ((data["NAME_EDUCATION_TYPE_Academic_degree"])))) + (np.maximum(((((((((np.maximum(((np.maximum(((data["ORGANIZATION_TYPE_Business_Entity_Type_2"])), ((data["NAME_EDUCATION_TYPE_Higher_education"]))))),
                                                                                                                               ((data["ORGANIZATION_TYPE_Realtor"])))) + (data["NAME_INCOME_TYPE_Maternity_leave"])) / 2.0)) + (data["NAME_INCOME_TYPE_Unemployed"])) / 2.0))), ((data["NAME_TYPE_SUITE_Children"]))))))
    v["i79"] = 0.095700 * np.tanh(((np.maximum(((np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"] > 0, data["FONDKAPREMONT_MODE_reg_oper_spec_account"], data["ORGANIZATION_TYPE_Industry__type_1"]))), ((
        data["ORGANIZATION_TYPE_Restaurant"])))) - (np.maximum((((((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) + (data["FONDKAPREMONT_MODE_reg_oper_spec_account"])) / 2.0))), ((((data["ORGANIZATION_TYPE_Industry__type_1"]) / 2.0)))))))
    v["i80"] = 0.099999 * np.tanh(((((data["NAME_INCOME_TYPE_Unemployed"]) + ((((((((data["ORGANIZATION_TYPE_Mobile"]) + ((((data["OCCUPATION_TYPE_Sales_staff"]) + (data["NAME_INCOME_TYPE_Unemployed"])) / 2.0))
                                                                                    ) / 2.0)) - (data["ORGANIZATION_TYPE_Industry__type_12"]))) - ((((data["ORGANIZATION_TYPE_Hotel"]) + (data["ORGANIZATION_TYPE_Trade__type_6"])) / 2.0)))))) / 2.0))
    v["i81"] = 0.098999 * np.tanh((((((-1.0 * ((((((((data["EXT_SOURCE_3"]) > (((((((data["NAME_EDUCATION_TYPE_Academic_degree"]) + ((((data["EXT_SOURCE_3"]) < (
        data["REG_CITY_NOT_WORK_CITY"])) * 1.))) / 2.0)) + (data["OCCUPATION_TYPE_Waiters_barmen_staff"])) / 2.0))) * 1.)) < (data["EXT_SOURCE_3"])) * 1.))))) * 2.0)) * 2.0))
    v["i82"] = 0.099260 * np.tanh(np.where(data["EXT_SOURCE_3"] < -99998, ((data["EXT_SOURCE_3"]) - (data["CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN"])), np.where(
        data["EXT_SOURCE_3"] > 0, data["CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN"], ((((1.74202001094818115)) < (((((data["EXT_SOURCE_3"]) * (data["EXT_SOURCE_3"]))) / 2.0))) * 1.))))
    v["i83"] = 0.093660 * np.tanh(((((((((((data["OCCUPATION_TYPE_Laborers"]) / 2.0)) < (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])) * 1.)) * 2.0)) + (
        ((np.minimum(((((((data["OCCUPATION_TYPE_Laborers"]) - (data["NAME_INCOME_TYPE_Working"]))) - (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])))), ((1.0)))) / 2.0))) / 2.0))
    v["i84"] = 0.099990 * np.tanh(((((data["NEW_SCORES_STD"]) * 2.0)) * (((((data["EXT_SOURCE_3"]) * 2.0)) - (np.minimum(((np.where(
        data["EXT_SOURCE_3"] < -99998, ((data["EXT_SOURCE_3"]) * 2.0), (-1.0 * ((data["NEW_SCORES_STD"])))))), ((data["NEW_CREDIT_TO_INCOME_RATIO"]))))))))
    v["i85"] = 0.099710 * np.tanh((((((((data["NEW_SCORES_STD"]) * (data["NEW_SCORES_STD"]))) * (
        ((((np.tanh((data["EXT_SOURCE_3"]))) * (data["NEW_SCORES_STD"]))) * 2.0)))) + (((data["NEW_SCORES_STD"]) * (data["NEW_SCORES_STD"])))) / 2.0))
    v["i86"] = 0.099950 * np.tanh(np.where(data["NEW_INC_PER_CHLD"] > 0, ((((((data["REGION_POPULATION_RELATIVE"]) < (data["LIVINGAREA_AVG"])) * 1.)) > (np.where(data["LIVINGAREA_AVG"] < -99998, data[
                                  "NEW_INC_PER_CHLD"], data["REGION_POPULATION_RELATIVE"]))) * 1.), np.where(data["LIVINGAREA_MODE"] < -99998, data["NEW_INC_PER_CHLD"], data["REGION_POPULATION_RELATIVE"])))
    v["i87"] = 0.096000 * np.tanh(((np.where(data["YEARS_BUILD_AVG"] < 0, data["OCCUPATION_TYPE_Waiters_barmen_staff"], (-1.0 * (
        (((data["LIVINGAPARTMENTS_MEDI"]) + (np.where((((data["YEARS_BUILD_MEDI"]) > (data["LIVINGAPARTMENTS_MEDI"])) * 1.) > 0, data["YEARS_BUILD_MEDI"], -2.0)))))))) / 2.0))
    v["i88"] = 0.099950 * np.tanh(((((((((((((-1.0 * ((((data["YEARS_BUILD_MODE"]) - (np.minimum(((1.0)), ((data["YEARS_BUILD_AVG"]))))))))) * 2.0)) * 2.0))
                                        * 2.0)) + (np.maximum(((data["NAME_INCOME_TYPE_Maternity_leave"])), ((data["ORGANIZATION_TYPE_Agriculture"]))))) / 2.0)) * 2.0))
    v["i89"] = 0.095039 * np.tanh(np.where(data["REG_REGION_NOT_LIVE_REGION"] > 0, (-1.0 * (((((data["CNT_CHILDREN"]) + (1.570796)) / 2.0)))), np.where(
        data["CNT_CHILDREN"] < 0, (-1.0 * (((((data["NAME_EDUCATION_TYPE_Academic_degree"]) + ((-1.0 * ((data["FONDKAPREMONT_MODE_not_specified"]))))) / 2.0)))), 0.318310)))
    v["i90"] = 0.100000 * np.tanh((-1.0 * ((((np.minimum(((data["INSTAL_PAYMENT_DIFF_VAR"])), ((((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]) + (data["INSTAL_PAYMENT_DIFF_VAR"])))))) + (
        np.maximum((((((((data["INSTAL_PAYMENT_DIFF_VAR"]) * 2.0)) > (data["INSTAL_PAYMENT_DIFF_STD"])) * 1.))), ((data["INSTAL_PAYMENT_DIFF_VAR"])))))))))
    v["i91"] = 0.099996 * np.tanh(((((np.where(data["INSTAL_DPD_MEAN"] < 0, ((data["INSTAL_DPD_MEAN"]) + (((0.318310) + (
        ((3.141593) * (data["INSTAL_DPD_MEAN"])))))), (((3.99870252609252930)) - (data["INSTAL_DPD_MEAN"])))) * 2.0)) * 2.0))
    v["i92"] = 0.079801 * np.tanh((((np.minimum(((data["CLOSED_CREDIT_DAY_OVERDUE_MEAN"])), ((data["CLOSED_AMT_CREDIT_SUM_SUM"])))) + (np.maximum(
        (((((((((data["NAME_INCOME_TYPE_Maternity_leave"]) * ((10.0)))) > (data["CLOSED_AMT_CREDIT_SUM_SUM"])) * 1.)) * (data["CLOSED_MONTHS_BALANCE_SIZE_SUM"])))), (((-1.0 * ((data["CLOSED_MONTHS_BALANCE_SIZE_SUM"])))))))) / 2.0))
    v["i93"] = 0.081035 * np.tanh(((((((data["ORGANIZATION_TYPE_Other"]) > (1.570796)) * 1.)) + (((((((data["NEW_CREDIT_TO_INCOME_RATIO"]) * 2.0)) * (
        (((1.570796) < ((((data["NEW_CREDIT_TO_INCOME_RATIO"]) + (-1.0)) / 2.0))) * 1.)))) - (data["FLAG_CONT_MOBILE"])))) / 2.0))
    v["i94"] = 0.093995 * np.tanh(((((((((((data["LIVE_CITY_NOT_WORK_CITY"]) * (data["ORGANIZATION_TYPE_Transport__type_1"]))) - (data["NAME_EDUCATION_TYPE_Academic_degree"]))) * 2.0)) * 2.0)) + (
        np.maximum(((np.maximum(((data["ORGANIZATION_TYPE_Realtor"])), ((data["OCCUPATION_TYPE_Secretaries"]))))), ((data["NAME_EDUCATION_TYPE_Academic_degree"]))))))
    v["i95"] = 0.080997 * np.tanh(((((1.0) + (data["NAME_CONTRACT_TYPE_Cash_loans"]))) * (np.maximum(((data["CLOSED_AMT_ANNUITY_MAX"])), ((
        np.where(((data["NAME_INCOME_TYPE_Student"]) - (data["CLOSED_AMT_ANNUITY_MAX"])) < 0, 0.636620, data["ORGANIZATION_TYPE_Transport__type_4"])))))))
    v["i96"] = 0.099801 * np.tanh(np.where(np.maximum((((((data["OBS_60_CNT_SOCIAL_CIRCLE"]) + (np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"] < 0, data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"], data[
                                  "DEF_30_CNT_SOCIAL_CIRCLE"]))) / 2.0))), ((data["DEF_30_CNT_SOCIAL_CIRCLE"]))) > 0, 3.141593, (-1.0 * (((((data["OBS_60_CNT_SOCIAL_CIRCLE"]) > (data["DEF_60_CNT_SOCIAL_CIRCLE"])) * 1.))))))
    v["i97"] = 0.096995 * np.tanh(np.where(data["CC_AMT_PAYMENT_TOTAL_CURRENT_VAR"] < -99998, (0.11329653859138489), (-1.0 * (
        (np.where(((data["CC_AMT_PAYMENT_TOTAL_CURRENT_VAR"]) * 2.0) < 0, ((data["CC_AMT_PAYMENT_TOTAL_CURRENT_VAR"]) * 2.0), (5.0)))))))
    v["i98"] = 0.097103 * np.tanh(np.where(data["DEF_60_CNT_SOCIAL_CIRCLE"] > 0, (-1.0 * ((data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"]))), ((((data["DEF_60_CNT_SOCIAL_CIRCLE"]) / 2.0)) + (
        (((np.maximum(((data["OBS_60_CNT_SOCIAL_CIRCLE"])), (((((data["DEF_60_CNT_SOCIAL_CIRCLE"]) < (data["OBS_30_CNT_SOCIAL_CIRCLE"])) * 1.))))) < (data["OBS_30_CNT_SOCIAL_CIRCLE"])) * 1.)))))
    v["i99"] = 0.100000 * np.tanh(((((np.tanh((np.minimum(((np.tanh((np.tanh((np.tanh(
        (((((((data["AMT_ANNUITY"]) * 2.0)) * 2.0)) * 2.0))))))))), ((0.636620)))))) - (data["EXT_SOURCE_2"]))) * ((4.0))))
    v["i100"] = 0.099095 * np.tanh(((np.tanh((data["AMT_CREDIT"]))) * (np.where(np.where(data["NAME_CONTRACT_TYPE_Revolving_loans"]
                                                                                         < 0, data["EXT_SOURCE_3"], 0.636620) < -99998, (-1.0 * ((data["AMT_CREDIT"]))), data["EXT_SOURCE_3"]))))
    v["i101"] = 0.064501 * np.tanh((((((data["REGION_RATING_CLIENT"]) * (data["DAYS_BIRTH"]))) + (((np.where(data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"] > 0,
                                                                                                             ((-2.0) - (data["DAYS_BIRTH"])), (((data["REGION_POPULATION_RELATIVE"]) > (data["DAYS_BIRTH"])) * 1.))) - (data["REGION_RATING_CLIENT"])))) / 2.0))
    v["i102"] = 0.098001 * np.tanh(((np.maximum((((-1.0 * ((data["DAYS_BIRTH"]))))), ((np.where(data["NEW_SCORES_STD"] < 0, data["DAYS_BIRTH"], data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]))))) * (
        (-1.0 * ((np.maximum((((((data["ORGANIZATION_TYPE_Transport__type_1"]) + (data["DAYS_BIRTH"])) / 2.0))), ((data["NAME_HOUSING_TYPE_Office_apartment"])))))))))
    v["i103"] = 0.094951 * np.tanh(np.where(((data["ORGANIZATION_TYPE_Mobile"]) * (data["NEW_SCORES_STD"])) > 0, ((data["WALLSMATERIAL_MODE_Wooden"]) * (data["NAME_EDUCATION_TYPE_Higher_education"])),
                                            (((((data["NAME_EDUCATION_TYPE_Higher_education"]) / 2.0)) + ((((data["NAME_INCOME_TYPE_Maternity_leave"]) + ((-1.0 * ((data["NEW_SCORES_STD"]))))) / 2.0))) / 2.0)))
    v["i104"] = 0.077000 * np.tanh(((np.where(data["ORGANIZATION_TYPE_Mobile"] < -99998, data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"], np.where(data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"] < -99998, data[
                                   "WEEKDAY_APPR_PROCESS_START_TUESDAY"], (((data["FLAG_DOCUMENT_3"]) + (((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) * (((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) - (3.0)))))) / 2.0)))) / 2.0))
    v["i105"] = 0.098001 * np.tanh((((((((((data["ORGANIZATION_TYPE_Transport__type_1"]) * (data["NAME_TYPE_SUITE_Spouse__partner"]))) + (
        ((data["ORGANIZATION_TYPE_Mobile"]) * (((np.maximum(((data["NAME_TYPE_SUITE_Spouse__partner"])), ((data["AMT_CREDIT"])))) * (data["AMT_CREDIT"])))))) / 2.0)) * 2.0)) * 2.0))
    v["i106"] = 0.098189 * np.tanh(((((-1.0 * ((data["INSTAL_PAYMENT_DIFF_VAR"])))) + (((np.where(data["ORGANIZATION_TYPE_Legal_Services"] > 0, np.where(data["INSTAL_AMT_PAYMENT_STD"] > 0, (-1.0 * (
        (data["ORGANIZATION_TYPE_Legal_Services"]))), data["ORGANIZATION_TYPE_Legal_Services"]), data["INSTAL_AMT_PAYMENT_STD"])) - (np.tanh((data["INSTAL_PAYMENT_DIFF_STD"])))))) / 2.0))
    v["i107"] = 0.098970 * np.tanh(((((data["ORGANIZATION_TYPE_Industry__type_4"]) + ((((data["ORGANIZATION_TYPE_Advertising"]) + (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) / 2.0))) / 2.0)))) + (
        np.minimum((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < (-1.0)) * 1.))), ((((2.0) - (data["AMT_ANNUITY"]))))))))
    v["i108"] = 0.094776 * np.tanh(((np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] < 0, np.where(data["REGION_RATING_CLIENT"] > 0, data["NEW_ANNUITY_TO_INCOME_RATIO"], data[
                                   "NEW_INC_PER_CHLD"]), np.where(data["AMT_ANNUITY"] < 0, data["AMT_ANNUITY"], (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (((data["AMT_ANNUITY"]) / 2.0))) * 1.)))) / 2.0))
    v["i109"] = 0.099494 * np.tanh(np.minimum(((np.maximum(((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"])), (((((-1.0 * ((np.tanh((np.maximum(((data["NEW_CREDIT_TO_INCOME_RATIO"])), ((data["AMT_CREDIT"]))))))))) / 2.0)))))),
                                              (((-1.0 * ((np.where(data["NEW_CREDIT_TO_INCOME_RATIO"] > 0, data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"], data["CLOSED_MONTHS_BALANCE_MIN_MIN"]))))))))
    v["i110"] = 0.097330 * np.tanh(np.where((((((data["CLOSED_AMT_ANNUITY_MEAN"]) + (data["AMT_ANNUITY"]))) > (0.636620)) * 1.) > 0, 2.0, np.where(
        ((np.tanh((3.0))) - (data["AMT_ANNUITY"])) > 0, data["NAME_INCOME_TYPE_Maternity_leave"], data["CLOSED_AMT_ANNUITY_MEAN"])))
    v["i111"] = 0.099758 * np.tanh(np.minimum((((-1.0 * ((data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"]))))), ((((((np.minimum(((data["NAME_INCOME_TYPE_Maternity_leave"])), ((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"])))) - (
        data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"]))) * (((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]) + (((data["CLOSED_MONTHS_BALANCE_MAX_MAX"]) + (data["NEW_CREDIT_TO_INCOME_RATIO"]))))))))))
    v["i112"] = 0.099646 * np.tanh((-1.0 * (((((data["ACTIVE_MONTHS_BALANCE_MIN_MIN"]) > ((((np.maximum(((0.636620)), (((((data["ACTIVE_MONTHS_BALANCE_MIN_MIN"]) < (
        (-1.0 * ((data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"]))))) * 1.))))) + ((((data["ACTIVE_MONTHS_BALANCE_MIN_MIN"]) > (0.636620)) * 1.))) / 2.0))) * 1.)))))
    v["i113"] = 0.097999 * np.tanh(np.tanh((((np.minimum(((((1.0) + (data["NEW_CREDIT_TO_INCOME_RATIO"])))), ((
        ((((((1.0) + (data["NEW_CREDIT_TO_INCOME_RATIO"]))) + (data["NEW_CREDIT_TO_INCOME_RATIO"]))) * (data["AMT_ANNUITY"])))))) * 2.0))))
    v["i114"] = 0.099820 * np.tanh((-1.0 * ((((np.where(data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"] > 0, (((0.318310) > (data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"])) * 1.), np.where(
        data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"] < -99998, data["NAME_INCOME_TYPE_Student"], np.maximum(((data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"])), ((data["AMT_CREDIT"])))))) * 2.0)))))
    v["i115"] = 0.099100 * np.tanh(np.where(((((data["NEW_DOC_IND_KURT"]) * 2.0)) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0))) < 0, ((
        data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0), (((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)) * 2.0)) < (data["NAME_CONTRACT_TYPE_Cash_loans"])) * 1.)))
    v["i116"] = 0.097027 * np.tanh(((((((((data["REFUSED_APP_CREDIT_PERC_MEAN"]) > (data["REFUSED_DAYS_DECISION_MEAN"])) * 1.)) > (np.minimum(((data["REFUSED_APP_CREDIT_PERC_MAX"])), ((
        data["REFUSED_DAYS_DECISION_MAX"]))))) * 1.)) * (((np.where(data["REFUSED_APP_CREDIT_PERC_MEAN"] > 0, -1.0, data["REFUSED_DAYS_DECISION_MAX"])) - (data["REFUSED_DAYS_DECISION_MAX"])))))
    v["i117"] = 0.099221 * np.tanh(((np.where(((((((data["NAME_INCOME_TYPE_Maternity_leave"]) < (data["NEW_CREDIT_TO_INCOME_RATIO"])) * 1.)) + (data["REFUSED_DAYS_DECISION_MEAN"])) / 2.0)
                                              > 0, ((data["DAYS_BIRTH"]) + (data["NEW_CREDIT_TO_INCOME_RATIO"])), (-1.0 * (((((data["DAYS_BIRTH"]) + (data["NAME_INCOME_TYPE_Student"])) / 2.0)))))) / 2.0))
    v["i118"] = 0.071701 * np.tanh(np.where(np.where(data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"] < 0, data["REFUSED_RATE_DOWN_PAYMENT_MAX"], data["REFUSED_APP_CREDIT_PERC_MEAN"]) < -99998, data[
                                   "NAME_INCOME_TYPE_Maternity_leave"], (-1.0 * ((np.maximum(((data["REFUSED_APP_CREDIT_PERC_MEAN"])), (((((10.68912410736083984)) + (data["CLOSED_MONTHS_BALANCE_MAX_MAX"]))))))))))
    v["i119"] = 0.099998 * np.tanh(np.maximum(((((np.where(data["AMT_CREDIT"] < 0, data["FLAG_WORK_PHONE"], (-1.0 * ((data["FLAG_WORK_PHONE"]))))) / 2.0))),
                                              ((np.where(((data["AMT_CREDIT"]) + (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])) < 0, -2.0, data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"])))))
    v["i120"] = 0.065500 * np.tanh(((((data["OCCUPATION_TYPE_Managers"]) + (((data["FLAG_DOCUMENT_3"]) / 2.0)))) * ((((data["NEW_LIVE_IND_SUM"]) + (np.where(
        ((data["OCCUPATION_TYPE_Managers"]) + (((data["FLAG_WORK_PHONE"]) / 2.0))) < 0, data["NAME_INCOME_TYPE_Student"], data["NAME_CONTRACT_TYPE_Cash_loans"]))) / 2.0))))
    v["i121"] = 0.099190 * np.tanh(np.where(np.minimum(((data["NEW_SCORES_STD"])), (((-1.0 * ((data["NAME_EDUCATION_TYPE_Higher_education"])))))) < 0, ((((data["NAME_EDUCATION_TYPE_Academic_degree"]) - (
        data["NAME_EDUCATION_TYPE_Higher_education"]))) * ((((data["NAME_CONTRACT_TYPE_Revolving_loans"]) + (data["NAME_INCOME_TYPE_Student"])) / 2.0))), data["OCCUPATION_TYPE_Core_staff"]))
    v["i122"] = 0.098898 * np.tanh((((((np.minimum((((((1.0) < (data["AMT_INCOME_TOTAL"])) * 1.))), ((((((data["NAME_CONTRACT_TYPE_Cash_loans"]) * 2.0)) * 2.0))))) + (
        data["NAME_CONTRACT_TYPE_Cash_loans"])) / 2.0)) * (((((data["AMT_INCOME_TOTAL"]) - (data["ORGANIZATION_TYPE_Bank"]))) * 2.0))))
    v["i123"] = 0.099904 * np.tanh(np.where(((((np.minimum(((data["ORGANIZATION_TYPE_Kindergarten"])), ((0.636620)))) + (data["NEW_CREDIT_TO_INCOME_RATIO"]))) - (
        1.570796)) > 0, ((data["ORGANIZATION_TYPE_Kindergarten"]) + (data["FLAG_WORK_PHONE"])), ((data["FLAG_WORK_PHONE"]) * (data["ORGANIZATION_TYPE_Kindergarten"]))))
    v["i124"] = 0.075030 * np.tanh(np.where((((data["NEW_CREDIT_TO_INCOME_RATIO"]) < (((((((data["NEW_CREDIT_TO_INCOME_RATIO"]) < (data["NAME_INCOME_TYPE_Maternity_leave"])) * 1.)) > (
        data["FLAG_EMAIL"])) * 1.))) * 1.) > 0, data["NAME_INCOME_TYPE_Unemployed"], (-1.0 * (((((data["FLAG_EMAIL"]) + (data["NAME_TYPE_SUITE_Group_of_people"])) / 2.0))))))
    v["i125"] = 0.099496 * np.tanh(((((((((((((data["NEW_CREDIT_TO_INCOME_RATIO"]) + (data["ORGANIZATION_TYPE_Transport__type_1"])) / 2.0)) - (3.0))) > (np.tanh((-2.0)))) * 1.)) * 2.0)) - (
        np.where(data["ORGANIZATION_TYPE_Transport__type_1"] > 0, data["ORGANIZATION_TYPE_Transport__type_1"], data["NAME_EDUCATION_TYPE_Academic_degree"]))))
    v["i126"] = 0.089995 * np.tanh(((data["NAME_INCOME_TYPE_Maternity_leave"]) - (np.maximum(((np.maximum(((((data["NAME_INCOME_TYPE_Student"]) - (data["NAME_TYPE_SUITE_Group_of_people"])))), ((
        data["NAME_TYPE_SUITE_Group_of_people"]))))), (((((((data["ORGANIZATION_TYPE_Industry__type_5"]) + (data["NAME_INCOME_TYPE_Student"]))) + (data["ORGANIZATION_TYPE_Kindergarten"])) / 2.0)))))))
    v["i127"] = 0.098800 * np.tanh((((((((np.maximum(((data["ORGANIZATION_TYPE_Telecom"])), (((((((0.318310) / 2.0)) + (((data["ORGANIZATION_TYPE_Electricity"]) + (0.318310)))) / 2.0)))))
                                         * (data["NEW_ANNUITY_TO_INCOME_RATIO"]))) + (data["ORGANIZATION_TYPE_Telecom"])) / 2.0)) - (data["OCCUPATION_TYPE_IT_staff"])))
    v["i128"] = 0.093700 * np.tanh(np.where(((data["NEW_SOURCES_PROD"]) + (1.0)) > 0, ((((np.minimum(((0.318310)), ((
        ((1.0) + (data["NEW_SOURCES_PROD"])))))) + (data["NEW_SOURCES_PROD"]))) * 2.0), data["ORGANIZATION_TYPE_Legal_Services"]))
    v["i129"] = 0.094999 * np.tanh((-1.0 * ((((((data["NAME_HOUSING_TYPE_Rented_apartment"]) * (((data["NAME_FAMILY_STATUS_Single___not_married"]) + (
        ((((data["ORGANIZATION_TYPE_Construction"]) + (((data["WEEKDAY_APPR_PROCESS_START_WEDNESDAY"]) - (data["ORGANIZATION_TYPE_Trade__type_4"]))))) + (data["WALLSMATERIAL_MODE_Others"]))))))) / 2.0)))))
    v["i130"] = 0.099065 * np.tanh(np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, ((((0.318310) - (
        (((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (2.0))) < (0.636620)) * 1.)))) * 2.0), (((-2.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) / 2.0)))
    v["i131"] = 0.081300 * np.tanh(np.minimum(((((data["NEW_DOC_IND_KURT"]) * (data["NEW_DOC_IND_KURT"])))), (((((((((data["NAME_CONTRACT_TYPE_Cash_loans"]) * (
        data["NEW_CREDIT_TO_INCOME_RATIO"]))) + (((((data["NEW_CREDIT_TO_INCOME_RATIO"]) * (data["NEW_CREDIT_TO_INCOME_RATIO"]))) + (data["NEW_DOC_IND_KURT"])))) / 2.0)) * 2.0)))))
    v["i132"] = 0.098000 * np.tanh((-1.0 * ((((((((np.where(data["CLOSED_MONTHS_BALANCE_MAX_MAX"] < 0, data["BURO_MONTHS_BALANCE_MAX_MAX"], data["CLOSED_MONTHS_BALANCE_MAX_MAX"])) + (
        data["BURO_STATUS_0_MEAN_MEAN"])) / 2.0)) > ((((0.318310) < (np.where(data["CLOSED_MONTHS_BALANCE_MAX_MAX"] < 0, 3.141593, data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]))) * 1.))) * 1.)))))
    v["i133"] = 0.034700 * np.tanh(np.minimum(((np.where(data["CLOSED_DAYS_CREDIT_MEAN"] < -99998, data["NAME_INCOME_TYPE_Maternity_leave"], (((data["CLOSED_DAYS_CREDIT_MEAN"]) < (data["CLOSED_AMT_CREDIT_SUM_SUM"])) * 1.)))),
                                              ((((np.tanh((np.where(data["CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN"] < 0, data["CLOSED_DAYS_CREDIT_MEAN"], data["CLOSED_AMT_CREDIT_SUM_SUM"])))) * (data["CLOSED_AMT_CREDIT_SUM_SUM"]))))))
    v["i134"] = 0.099900 * np.tanh(((np.where((-1.0 * ((((((((3.0) + (data["CLOSED_DAYS_CREDIT_MEAN"])) / 2.0)) < (data["CLOSED_AMT_CREDIT_SUM_SUM"])) * 1.))))
                                              < 0, -2.0, (((2.0) < (np.maximum(((data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"])), ((data["CLOSED_DAYS_CREDIT_MEAN"]))))) * 1.))) * 2.0))
    v["i135"] = 0.098300 * np.tanh(((np.where((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) > (0.318310)) * 1.) > 0, data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"], ((
        (data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) > ((((((data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]) + (0.318310)) / 2.0)) / 2.0))) * 1.))) * (3.0)))
    v["i136"] = 0.097500 * np.tanh((-1.0 * (((((np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"] < 0, data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"], data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])) > (
        (((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) + ((((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) < (np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"] < 0, data["NEW_CREDIT_TO_INCOME_RATIO"], 3.141593))) * 1.))) / 2.0))) * 1.)))))
    v["i137"] = 0.098499 * np.tanh(((((np.where(data["ACTIVE_AMT_CREDIT_SUM_MEAN"] < -99998, data["ACTIVE_AMT_CREDIT_SUM_MEAN"], ((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]) - (
        data["ACTIVE_AMT_CREDIT_SUM_MEAN"]))) * 2.0))) - (data["ACTIVE_AMT_CREDIT_SUM_MEAN"]))) - (np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"] < 0, 0.318310, data["ACTIVE_AMT_CREDIT_SUM_MEAN"]))))
    v["i138"] = 0.098411 * np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"] > 0, (((((data["ACTIVE_DAYS_CREDIT_VAR"]) < (-2.0)) * 1.)) + (
        data["AMT_CREDIT"])), (((data["ACTIVE_DAYS_CREDIT_VAR"]) > (np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"] < -99998, -2.0, 3.141593))) * 1.)))
    v["i139"] = 0.095000 * np.tanh(np.where(data["ACTIVE_DAYS_CREDIT_VAR"] > 0, data["BURO_DAYS_CREDIT_VAR"], np.where(np.minimum(((data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"])), ((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]))) < -99998, ((
        (data["ACTIVE_AMT_CREDIT_SUM_SUM"]) > (data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"])) * 1.), np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"] < 0, data["ACTIVE_AMT_CREDIT_SUM_SUM"], data["BURO_DAYS_CREDIT_VAR"]))))
    v["i140"] = 0.072769 * np.tanh((((((np.maximum(((np.maximum(((((data["BURO_DAYS_CREDIT_VAR"]) + (data["NAME_INCOME_TYPE_Maternity_leave"])))), ((((data["CLOSED_DAYS_CREDIT_VAR"]) * 2.0)))))),
                                                   ((2.0)))) + (((data["ACTIVE_DAYS_CREDIT_VAR"]) - (data["BURO_DAYS_CREDIT_VAR"]))))) < (data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"])) * 1.))
    v["i141"] = 0.099500 * np.tanh(np.where(data["CLOSED_AMT_CREDIT_SUM_DEBT_MEAN"] < 0, (((-2.0) > (np.where(data["CLOSED_AMT_CREDIT_SUM_DEBT_MEAN"] < -99998, ((
        ((0.318310) + (data["ACTIVE_DAYS_CREDIT_VAR"]))) * (data["CLOSED_AMT_CREDIT_SUM_DEBT_MEAN"])), data["BURO_DAYS_CREDIT_VAR"]))) * 1.), ((data["CLOSED_DAYS_CREDIT_VAR"]) * 2.0)))
    v["i142"] = 0.090700 * np.tanh((((((((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) > (3.0)) * 1.)) - ((((data["ACTIVE_DAYS_CREDIT_VAR"]) > (np.where(
        data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"] < 0, 3.141593, (((data["CLOSED_AMT_CREDIT_SUM_MEAN"]) < ((-1.0 * ((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]))))) * 1.)))) * 1.)))) * 2.0))
    v["i143"] = 0.099900 * np.tanh((((-1.0 * ((((((((1.46526610851287842)) - (np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, data["CLOSED_DAYS_CREDIT_MEAN"], (1.46526610851287842)))))
                                                 < (np.maximum(((data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"])), (((((3.141593) < (data["CLOSED_AMT_CREDIT_SUM_MEAN"])) * 1.)))))) * 1.))))) * 2.0))
    v["i144"] = 0.071750 * np.tanh(np.maximum(((((np.where(data["REG_CITY_NOT_LIVE_CITY"] < 0, ((((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) + (data["BURO_DAYS_CREDIT_VAR"]))) * (
        data["NAME_INCOME_TYPE_Unemployed"])), data["CLOSED_AMT_CREDIT_SUM_DEBT_MEAN"])) * (data["CLOSED_AMT_CREDIT_SUM_DEBT_MEAN"])))), (((((1.570796) < (data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"])) * 1.)))))
    v["i145"] = 0.095900 * np.tanh(np.where(((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) - (3.0)) > 0, ((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) - (3.0)), ((
        data["NAME_INCOME_TYPE_Maternity_leave"]) - ((((np.maximum(((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"])), ((data["ORGANIZATION_TYPE_Police"])))) > (0.318310)) * 1.)))))
    v["i146"] = 0.099969 * np.tanh(np.where(np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"], data["ORGANIZATION_TYPE_Housing"]) > 0, np.where(
        data["ORGANIZATION_TYPE_Housing"] > 0, data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"], ((3.0) - (data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]))), (((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) > (1.570796)) * 1.)))
    v["i147"] = 0.089499 * np.tanh(((data["HOUSETYPE_MODE_specific_housing"]) * (np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < 0, np.maximum((((((data["ACTIVE_DAYS_CREDIT_VAR"]) > (-1.0)) * 1.))),
                                                                                                                                         ((data["REG_CITY_NOT_WORK_CITY"]))), (((((data["ACTIVE_DAYS_CREDIT_VAR"]) + (((data["REG_CITY_NOT_WORK_CITY"]) * 2.0))) / 2.0)) * 2.0)))))
    v["i148"] = 0.068699 * np.tanh(((((((data["WALLSMATERIAL_MODE_Block"]) * (((data["ORGANIZATION_TYPE_Trade__type_4"]) * 2.0)))) - (data["NAME_EDUCATION_TYPE_Academic_degree"]))) - (
        np.where(data["WALLSMATERIAL_MODE_Block"] < 0, data["NAME_INCOME_TYPE_Student"], np.maximum(((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"])), ((data["NAME_INCOME_TYPE_Student"])))))))
    v["i149"] = 0.096025 * np.tanh(np.where(data["APPROVED_RATE_DOWN_PAYMENT_MEAN"] < -99998, 0.0, (((np.maximum(((data["APPROVED_RATE_DOWN_PAYMENT_MEAN"])), ((data["NAME_INCOME_TYPE_Unemployed"])))) + (
        np.where(data["APPROVED_RATE_DOWN_PAYMENT_MAX"] > 0, (((data["APPROVED_RATE_DOWN_PAYMENT_MEAN"]) < (data["APPROVED_RATE_DOWN_PAYMENT_MAX"])) * 1.), data["APPROVED_RATE_DOWN_PAYMENT_MEAN"]))) / 2.0)))
    v["i150"] = 0.094497 * np.tanh(np.where(data["NAME_HOUSING_TYPE_Office_apartment"] > 0, data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"], ((np.maximum(((((((np.where(data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"] < -99998, data[
                                   "ORGANIZATION_TYPE_Trade__type_3"], data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"])) / 2.0)) / 2.0))), ((data["ORGANIZATION_TYPE_Electricity"])))) * (data["ORGANIZATION_TYPE_Trade__type_3"]))))
    v["i151"] = 0.095500 * np.tanh(((((((((1.570796) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) * ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (-1.0)) * 1.)))) + (
        (((-1.0 * (((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > ((2.10143613815307617))) * 1.))))) * 2.0)))) * 2.0))
    v["i152"] = 0.099880 * np.tanh(np.where(data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"] < 0, (-1.0 * (((((((((data["REFUSED_DAYS_DECISION_MEAN"]) + (data["NEW_CREDIT_TO_INCOME_RATIO"]))) * 2.0)) > (
        ((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) * ((((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) + (data["REFUSED_AMT_CREDIT_MEAN"])) / 2.0))))) * 1.)))), data["NEW_CREDIT_TO_INCOME_RATIO"]))
    v["i153"] = 0.099369 * np.tanh((-1.0 * ((((((((3.141593) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) / 2.0)) < ((((2.0) + (np.where(
        (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (-1.0)) * 1.) > 0, ((1.570796) / 2.0), data["NAME_INCOME_TYPE_Maternity_leave"]))) / 2.0))) * 1.)))))
    v["i154"] = 0.100000 * np.tanh(((((((((((data["DEF_30_CNT_SOCIAL_CIRCLE"]) * 2.0)) + (((data["DEF_30_CNT_SOCIAL_CIRCLE"]) - (
        np.minimum(((data["DEF_60_CNT_SOCIAL_CIRCLE"])), ((3.0))))))) / 2.0)) - ((3.0)))) > (((((data["AMT_CREDIT"]) * 2.0)) * 2.0))) * 1.))
    v["i155"] = 0.049000 * np.tanh(np.minimum((((((2.0) + (data["NAME_CONTRACT_TYPE_Cash_loans"])) / 2.0))), ((
        (((np.where(data["NAME_CONTRACT_TYPE_Cash_loans"] > 0, data["NEW_DOC_IND_KURT"], 2.0)) < ((-1.0 * ((2.0))))) * 1.)))))
    v["i156"] = 0.059949 * np.tanh(np.where(data["REGION_POPULATION_RELATIVE"] < 0, (((((data["NAME_CONTRACT_TYPE_Revolving_loans"]) - (((data["AMT_INCOME_TOTAL"]) / 2.0)))) < (
        ((data["REGION_POPULATION_RELATIVE"]) * 2.0))) * 1.), (-1.0 * ((np.where(data["AMT_INCOME_TOTAL"] > 0, data["NAME_CONTRACT_TYPE_Revolving_loans"], data["ORGANIZATION_TYPE_Trade__type_5"]))))))
    v["i157"] = 0.098560 * np.tanh(np.where(np.tanh(((((((data["AMT_INCOME_TOTAL"]) < (0.318310)) * 1.)) - ((((data["REGION_RATING_CLIENT_W_CITY"]) < (
        data["ORGANIZATION_TYPE_Trade__type_4"])) * 1.))))) < 0, data["REGION_RATING_CLIENT_W_CITY"], (((0.318310) < ((-1.0 * ((data["REGION_RATING_CLIENT_W_CITY"]))))) * 1.)))
    v["i158"] = 0.019660 * np.tanh(np.where((((data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"]) > (3.141593)) * 1.) > 0, -2.0, np.where(data["AMT_INCOME_TOTAL"] < 0, data[
                                   "WALLSMATERIAL_MODE_Mixed"], ((((((data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"]) < ((-1.0 * ((data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"]))))) * 1.)) < (data["AMT_INCOME_TOTAL"])) * 1.))))
    v["i159"] = 0.090429 * np.tanh((((((((np.maximum(((1.570796)), ((data["AMT_ANNUITY"])))) + (((data["NEW_CREDIT_TO_INCOME_RATIO"]) / 2.0)))) < (
        np.maximum(((data["AMT_INCOME_TOTAL"])), (((((-1.0 * ((data["AMT_INCOME_TOTAL"])))) - (data["AMT_INCOME_TOTAL"]))))))) * 1.)) * 2.0))
    v["i160"] = 0.088213 * np.tanh(np.minimum(((((((((((data["AMT_INCOME_TOTAL"]) + (data["ORGANIZATION_TYPE_Trade__type_4"])) / 2.0)) / 2.0)) > (data["NEW_CREDIT_TO_INCOME_RATIO"])) * 1.))),
                                              (((((((((data["NEW_CREDIT_TO_INCOME_RATIO"]) - (data["ORGANIZATION_TYPE_Hotel"]))) - (data["ORGANIZATION_TYPE_Trade__type_4"]))) + (0.318310)) / 2.0)))))
    v["i161"] = 0.096950 * np.tanh(((((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, (((((3.0) / 2.0)) < ((-1.0 * ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))
                                                                                          * 1.), (((data["AMT_INCOME_TOTAL"]) < ((-1.0 * ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)))))) * 1.))) * 2.0)) * 2.0))
    v["i162"] = 0.080995 * np.tanh(((data["NAME_INCOME_TYPE_Maternity_leave"]) - ((((((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (1.0)) / 2.0)) > (
        0.636620)) * 1.)) - (np.maximum(((0.636620)), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))) > (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0))) * 1.))))
    v["i163"] = 0.082988 * np.tanh(((data["ORGANIZATION_TYPE_Security_Ministries"]) * ((-1.0 * ((np.where(data["AMT_ANNUITY"] > 0, data["AMT_INCOME_TOTAL"], np.where(
        data["AMT_INCOME_TOTAL"] > 0, 1.570796, (-1.0 * ((np.where(data["ORGANIZATION_TYPE_Security_Ministries"] > 0, data["AMT_ANNUITY"], data["NAME_INCOME_TYPE_Working"]))))))))))))
    v["i164"] = 0.030499 * np.tanh((((((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]) > (((data["AMT_INCOME_TOTAL"]) * (data["AMT_INCOME_TOTAL"])))) * 1.)) * (((np.where(
        data["APPROVED_RATE_DOWN_PAYMENT_MEAN"] > 0, data["AMT_INCOME_TOTAL"], (((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) / 2.0)) < (data["APPROVED_AMT_DOWN_PAYMENT_MAX"])) * 1.))) * 2.0))))
    v["i165"] = 0.099802 * np.tanh(((((np.where(((1.570796) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) < 0, 3.0, ((
        data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * ((((np.tanh((0.636620))) < (data["APPROVED_AMT_DOWN_PAYMENT_MAX"])) * 1.))))) * 2.0)) * 2.0))
    v["i166"] = 0.090097 * np.tanh(((((((((((-1.0 * ((np.minimum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))))) > (
        1.570796)) * 1.)) - ((((data["ORGANIZATION_TYPE_Industry__type_9"]) > (((2.0) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) * 1.)))) * 2.0)) * 2.0))
    v["i167"] = 0.099620 * np.tanh(np.where(((((-1.0 * ((0.636620)))) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) / 2.0) > 0, np.minimum(((0.318310)), ((
        data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]))), np.where(data["INSTAL_DAYS_ENTRY_PAYMENT_STD"] < 0, (((-1.0) > (data["INSTAL_DAYS_ENTRY_PAYMENT_STD"])) * 1.), -1.0)))
    v["i168"] = 0.099730 * np.tanh(((((np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"] > 0, ((np.maximum(((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"])), ((data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"])))) - (
        1.570796)), (((data["CC_CNT_DRAWINGS_OTHER_CURRENT_MAX"]) > (np.maximum(((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"])), ((data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"]))))) * 1.))) * 2.0)) * 2.0))
    v["i169"] = 0.096773 * np.tanh(np.where(data["APPROVED_AMT_DOWN_PAYMENT_MAX"] > 0, np.where(data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"] < 0, 0.318310, data["APPROVED_AMT_DOWN_PAYMENT_MAX"]),
                                            np.where(data["NAME_CONTRACT_TYPE_Cash_loans"] > 0, (-1.0 * ((data["NAME_INCOME_TYPE_Student"]))), ((((data["APPROVED_RATE_DOWN_PAYMENT_MAX"]) / 2.0)) / 2.0))))
    v["i170"] = 0.099999 * np.tanh(np.tanh((np.where(data["INSTAL_DAYS_ENTRY_PAYMENT_STD"] > 0, data["INSTAL_AMT_PAYMENT_STD"], (-1.0 * (
        (((((((np.tanh((((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]) / 2.0)))) > (data["INSTAL_AMT_PAYMENT_STD"])) * 1.)) + ((((data["INSTAL_AMT_PAYMENT_STD"]) + (data["INSTAL_DAYS_ENTRY_PAYMENT_STD"])) / 2.0))) / 2.0))))))))
    v["i171"] = 0.097000 * np.tanh(((((((data["PREV_RATE_DOWN_PAYMENT_MEAN"]) > (0.636620)) * 1.)) > (np.where((((np.tanh(((-1.0 * ((0.636620)))))) < (
        data["PREV_RATE_DOWN_PAYMENT_MEAN"])) * 1.) > 0, data["PREV_RATE_DOWN_PAYMENT_MEAN"], (-1.0 * ((data["PREV_AMT_DOWN_PAYMENT_MAX"])))))) * 1.))
    v["i172"] = 0.098900 * np.tanh(np.minimum(((np.where((((data["DAYS_REGISTRATION"]) < (data["DAYS_BIRTH"])) * 1.) > 0, ((data["DAYS_BIRTH"]) / 2.0), data[
                                   "NAME_EDUCATION_TYPE_Incomplete_higher"]))), (((((-1.0 * (((((data["DAYS_BIRTH"]) > (((3.0) / 2.0))) * 1.))))) * 2.0)))))
    v["i173"] = 0.075100 * np.tanh(((((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]) * (np.where((((data["NEW_LIVE_IND_SUM"]) + (
        data["DAYS_REGISTRATION"])) / 2.0) > 0, data["DAYS_BIRTH"], (((-2.0) > (data["DAYS_BIRTH"])) * 1.))))) + ((((data["DAYS_REGISTRATION"]) < (-2.0)) * 1.))))
    v["i174"] = 0.097899 * np.tanh(((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] > 0, ((data["DAYS_BIRTH"]) * (((data["AMT_ANNUITY"]) - (
        data["NEW_CREDIT_TO_ANNUITY_RATIO"])))), ((0.318310) + ((((data["DAYS_BIRTH"]) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) / 2.0))))) / 2.0))
    v["i175"] = 0.100000 * np.tanh(np.where(np.where(np.where(data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"] < -99998, data["DAYS_BIRTH"], data["NAME_EDUCATION_TYPE_Secondary___secondary_special"])
                                                     > 0, data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"], data["DAYS_BIRTH"]) > 0, (-1.0 * ((data["DAYS_BIRTH"]))), (((data["DAYS_BIRTH"]) < ((-1.0 * ((1.570796))))) * 1.)))
    v["i176"] = 0.098959 * np.tanh(np.where((((data["NEW_SOURCES_PROD"]) + (data["NAME_INCOME_TYPE_Pensioner"])) / 2.0) > 0, data["NAME_INCOME_TYPE_Pensioner"], np.where(data["NEW_SOURCES_PROD"] < -99998, np.minimum(
        ((data["NAME_INCOME_TYPE_Maternity_leave"])), (((-1.0 * (((((data["NAME_INCOME_TYPE_Pensioner"]) + (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])) / 2.0))))))), data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])))
    v["i177"] = 0.099800 * np.tanh((((((np.maximum(((((((data["EXT_SOURCE_3"]) * 2.0)) * (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])))), ((np.maximum(
        ((((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]) * (data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"])))), ((0.636620))))))) * (-2.0))) > (data["EXT_SOURCE_3"])) * 1.))
    v["i178"] = 0.100000 * np.tanh(np.where(data["CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN"] > 0, (2.04872536659240723), ((data["CLOSED_DAYS_CREDIT_MEAN"]) * ((((((data["CLOSED_DAYS_CREDIT_MEAN"]) > (
        (2.04872536659240723))) * 1.)) + ((((((data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"]) > ((2.04872536659240723))) * 1.)) * (data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"]))))))))
    v["i179"] = 0.099995 * np.tanh(np.where(data["EXT_SOURCE_3"] < 0, data["NAME_INCOME_TYPE_Maternity_leave"], (((((data["EXT_SOURCE_3"]) > (
        data["NAME_EDUCATION_TYPE_Higher_education"])) * 1.)) + (np.where(data["NAME_EDUCATION_TYPE_Secondary___secondary_special"] < 0, 0.318310, (-1.0 * ((data["EXT_SOURCE_3"]))))))))
    v["i180"] = 0.096002 * np.tanh(np.where(((((-1.0 * ((data["NAME_INCOME_TYPE_State_servant"])))) + (data["NAME_EDUCATION_TYPE_Secondary___secondary_special"])) / 2.0) > 0, np.minimum(((data["NAME_INCOME_TYPE_State_servant"])),
                                                                                                                                                                                          ((((data["ACTIVE_DAYS_CREDIT_VAR"]) * (data["NAME_INCOME_TYPE_State_servant"]))))), (((((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]) > (data["ACTIVE_DAYS_CREDIT_VAR"])) * 1.)) / 2.0)))
    v["i181"] = 0.080049 * np.tanh(np.minimum(((np.where(((((-1.0) / 2.0)) - (data["EXT_SOURCE_1"])) > 0, (((data["EXT_SOURCE_1"]) > (data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"]))
                                                                                                           * 1.), ((data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"]) / 2.0)))), (((((data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"]) < (data["EXT_SOURCE_1"])) * 1.)))))
    v["i182"] = 0.097095 * np.tanh(np.where(data["WALLSMATERIAL_MODE_Wooden"] > 0, data["ORGANIZATION_TYPE_Self_employed"], (((1.570796) < (
        np.where((-1.0 * ((data["ORGANIZATION_TYPE_Self_employed"]))) < 0, (((1.0)) - (data["DAYS_BIRTH"])), ((0.318310) - (data["DAYS_BIRTH"]))))) * 1.)))
    v["i183"] = 0.096999 * np.tanh(np.minimum((((-1.0 * (((((((data["NAME_FAMILY_STATUS_Widow"]) * 2.0)) + (data["NEW_EMPLOY_TO_BIRTH_RATIO"])) / 2.0)))))),
                                              ((np.maximum(((data["NEW_EMPLOY_TO_BIRTH_RATIO"])), (((((1.570796) < (((data["DAYS_EMPLOYED"]) * 2.0))) * 1.))))))))
    v["i184"] = 0.099950 * np.tanh((-1.0 * ((((((np.where(data["ORGANIZATION_TYPE_XNA"] > 0, (((data["EXT_SOURCE_3"]) > (data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"])) * 1.),
                                                          ((((-1.0 * (((((data["EXT_SOURCE_3"]) > (1.570796)) * 1.))))) < (data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"])) * 1.))) * 2.0)) * 2.0)))))
    v["i185"] = 0.099008 * np.tanh(np.where(data["EXT_SOURCE_3"] > 0, (-1.0 * ((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]))), (((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]) > (
        np.maximum((((((data["EXT_SOURCE_3"]) + ((((data["EXT_SOURCE_3"]) + ((5.0))) / 2.0))) / 2.0))), (((-1.0 * ((0.318310)))))))) * 1.)))
    v["i186"] = 0.099100 * np.tanh(((np.where(data["DAYS_BIRTH"] < 0, np.where(data["FLAG_EMP_PHONE"] < 0, (-1.0 * ((0.318310))), data[
                                   "DAYS_BIRTH"]), np.where(data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"] < 0, data["DAYS_BIRTH"], (-1.0 * ((0.0)))))) * (data["CODE_GENDER"])))
    v["i187"] = 0.092795 * np.tanh(np.tanh((np.tanh(((-1.0 * (((((((((np.tanh((0.318310))) * (np.tanh(
        (data["ORGANIZATION_TYPE_XNA"]))))) < (np.tanh((data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"])))) * 1.)) - (0.318310))))))))))
    v["i188"] = 0.099915 * np.tanh(((((((np.minimum(((((((data["INSTAL_DPD_MAX"]) * 2.0)) * 2.0))), ((((data["INSTAL_DAYS_ENTRY_PAYMENT_SUM"]) + (
        data["INSTAL_DBD_MAX"])))))) - (data["INSTAL_DBD_MEAN"]))) + (((data["INSTAL_DAYS_ENTRY_PAYMENT_SUM"]) + (data["INSTAL_DBD_MAX"]))))) * 2.0))
    v["i189"] = 0.099484 * np.tanh(np.maximum(((((np.where(data["APPROVED_AMT_DOWN_PAYMENT_MAX"] > 0, ((data["EXT_SOURCE_3"]) * 2.0), data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"])) + (
        data["NAME_EDUCATION_TYPE_Secondary___secondary_special"])))), ((((np.where(data["NAME_EDUCATION_TYPE_Secondary___secondary_special"] < 0, data["NEW_CREDIT_TO_INCOME_RATIO"], 0.318310)) * (0.636620))))))
    v["i190"] = 0.098000 * np.tanh(np.minimum(((np.where(data["APPROVED_AMT_GOODS_PRICE_MEAN"] > 0, ((((data["APPROVED_AMT_GOODS_PRICE_MAX"]) - (data["APPROVED_AMT_GOODS_PRICE_MEAN"]))) + (
        ((data["APPROVED_AMT_GOODS_PRICE_MAX"]) - ((-1.0 * ((data["PREV_PRODUCT_COMBINATION_POS_industry_with_interest_MEAN"]))))))), 0.318310))), (((-1.0 * ((data["PREV_PRODUCT_COMBINATION_POS_industry_with_interest_MEAN"])))))))
    v["i191"] = 0.074830 * np.tanh((-1.0 * (((((((((((((((((data["CC_AMT_PAYMENT_TOTAL_CURRENT_VAR"]) > (((data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"]) * 2.0))) * 1.)) + (
        data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"])) / 2.0)) > ((((2.0) < (data["CC_AMT_PAYMENT_CURRENT_VAR"])) * 1.))) * 1.)) * 2.0)) * 2.0)) * 2.0)))))
    v["i192"] = 0.095648 * np.tanh(np.where(data["OCCUPATION_TYPE_Sales_staff"] > 0, data["NAME_EDUCATION_TYPE_Higher_education"], np.tanh((np.where(data["NAME_TYPE_SUITE_Spouse__partner"] > 0, data[
                                   "OCCUPATION_TYPE_Sales_staff"], (((data["NAME_TYPE_SUITE_Spouse__partner"]) < (np.where(data["HOUSETYPE_MODE_block_of_flats"] > 0, data["NAME_EDUCATION_TYPE_Higher_education"], -1.0))) * 1.))))))
    v["i193"] = 0.097520 * np.tanh(np.where(data["APPROVED_RATE_DOWN_PAYMENT_MEAN"] < -99998, data["NAME_FAMILY_STATUS_Widow"], (((data["APPROVED_AMT_DOWN_PAYMENT_MEAN"]) > (
        ((((((((((np.maximum(((1.0)), ((data["APPROVED_RATE_DOWN_PAYMENT_MEAN"])))) + (data["NAME_INCOME_TYPE_Student"])) / 2.0)) * 2.0)) + (data["REG_REGION_NOT_LIVE_REGION"])) / 2.0)) * 2.0))) * 1.)))
    v["i194"] = 0.098200 * np.tanh((((((np.tanh((data["NEW_SCORES_STD"]))) > ((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"]) < (
        ((((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) > (1.0)) * 1.)) + ((((1.570796) < (data["NEW_SCORES_STD"])) * 1.))) / 2.0))) * 1.))) * 1.)) * ((4.11938047409057617))))
    v["i195"] = 0.099496 * np.tanh(((data["REFUSED_AMT_APPLICATION_MAX"]) * ((((data["REFUSED_AMT_APPLICATION_MAX"]) < (np.where(data["REFUSED_AMT_APPLICATION_MAX"] > 0, ((
        ((((1.0) + (data["REFUSED_AMT_APPLICATION_MEAN"])) / 2.0)) < (data["REFUSED_AMT_APPLICATION_MAX"])) * 1.), (((data["REFUSED_AMT_APPLICATION_MEAN"]) + (data["REFUSED_AMT_CREDIT_MEAN"])) / 2.0)))) * 1.))))
    v["i196"] = 0.099004 * np.tanh(((((((np.tanh((((0.636620) - (data["REFUSED_RATE_DOWN_PAYMENT_MAX"]))))) < (np.where(data["REG_REGION_NOT_LIVE_REGION"]
                                                                                                                        > 0, ((0.636620) - (data["NEW_SCORES_STD"])), data["NEW_SCORES_STD"]))) * 1.)) < (data["REFUSED_AMT_DOWN_PAYMENT_MEAN"])) * 1.))
    v["i197"] = 0.094900 * np.tanh((-1.0 * ((np.where((((1.0) < (data["NEW_SCORES_STD"])) * 1.) > 0, (((data["OCCUPATION_TYPE_Private_service_staff"]) + (
        (((((1.570796) + (data["REG_REGION_NOT_LIVE_REGION"]))) < (data["NEW_SCORES_STD"])) * 1.))) / 2.0), data["REG_REGION_NOT_LIVE_REGION"])))))
    v["i198"] = 0.100000 * np.tanh(np.where((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > ((6.18439579010009766))) * 1.) > 0, data["AMT_ANNUITY"], ((
        np.where(data["AMT_ANNUITY"] > 0, (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (1.570796)) * 1.), data["ORGANIZATION_TYPE_Trade__type_5"])) * (-1.0))))
    v["i199"] = 0.098500 * np.tanh(((data["NEW_CREDIT_TO_INCOME_RATIO"]) * (np.minimum((((((((data["ORGANIZATION_TYPE_Industry__type_6"]) - (data["YEARS_BUILD_MEDI"])))
                                                                                           < ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < ((4.81200885772705078))) * 1.))) * 1.))), ((((0.318310) - (data["YEARS_BUILD_MEDI"]))))))))
    v["i200"] = 0.066990 * np.tanh(np.where(data["YEARS_BUILD_AVG"] < 0, (-1.0 * (((((data["FONDKAPREMONT_MODE_org_spec_account"]) + (data["WALLSMATERIAL_MODE_Monolithic"])) / 2.0)))),
                                            np.where(((0.318310) - (np.tanh((data["YEARS_BUILD_AVG"])))) > 0, (-1.0 * ((data["FONDKAPREMONT_MODE_org_spec_account"]))), data["FONDKAPREMONT_MODE_org_spec_account"])))
    v["i201"] = 0.099948 * np.tanh(((((data["ORGANIZATION_TYPE_Legal_Services"]) + (((data["ORGANIZATION_TYPE_Industry__type_6"]) * (data["NAME_FAMILY_STATUS_Separated"]))))) + (
        (((0.636620) < (((data["NAME_INCOME_TYPE_Maternity_leave"]) + (((data["ORGANIZATION_TYPE_Legal_Services"]) + (((data["DAYS_REGISTRATION"]) / 2.0))))))) * 1.))))
    v["i202"] = 0.099788 * np.tanh((((-1.0 * ((np.where(data["CC_AMT_PAYMENT_TOTAL_CURRENT_VAR"] < 0, np.where(data["AMT_ANNUITY"] < 0, (((1.570796) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))
                                                                                                                                         * 1.), (((((data["CC_AMT_PAYMENT_TOTAL_CURRENT_VAR"]) * 2.0)) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)), data["CC_AMT_PAYMENT_TOTAL_CURRENT_VAR"]))))) * 2.0))
    v["i203"] = 0.081100 * np.tanh((((((((data["DAYS_REGISTRATION"]) < (np.minimum(((np.tanh((data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"])))), ((
        ((((((0.318310) < (data["CC_CNT_DRAWINGS_OTHER_CURRENT_MAX"])) * 1.)) + (np.minimum(((0.636620)), ((data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"]))))) / 2.0)))))) * 1.)) * 2.0)) * 2.0))
    v["i204"] = 0.005498 * np.tanh((-1.0 * (((((np.minimum(((data["CC_AMT_PAYMENT_CURRENT_MAX"])), ((data["CC_AMT_DRAWINGS_CURRENT_VAR"])))) > ((-1.0 * (
        ((((((((data["CC_AMT_PAYMENT_CURRENT_VAR"]) * (data["CC_AMT_PAYMENT_CURRENT_VAR"]))) > (data["CC_AMT_PAYMENT_CURRENT_MAX"])) * 1.)) / 2.0)))))) * 1.)))))
    v["i205"] = 0.098500 * np.tanh((((-1.0 * ((((((((np.where(data["CC_AMT_PAYMENT_CURRENT_MEAN"] < 0, data["NAME_INCOME_TYPE_Student"], ((
        (((data["CC_AMT_PAYMENT_TOTAL_CURRENT_VAR"]) * 2.0)) < (((data["NAME_INCOME_TYPE_Student"]) + (data["CC_AMT_PAYMENT_CURRENT_MEAN"])))) * 1.))) * 2.0)) * 2.0)) * 2.0))))) * 2.0))
    v["i206"] = 0.069950 * np.tanh(((np.tanh((np.tanh((np.where(np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] > 0, data["AMT_ANNUITY"], ((data["NEW_CREDIT_TO_INCOME_RATIO"]) + (
        data["NEW_ANNUITY_TO_INCOME_RATIO"]))) < 0, data["AMT_ANNUITY"], (((data["NAME_INCOME_TYPE_Student"]) > (data["AMT_ANNUITY"])) * 1.))))))) / 2.0))
    v["i207"] = 0.098931 * np.tanh(np.where(data["FONDKAPREMONT_MODE_org_spec_account"] > 0, ((data["NEW_ANNUITY_TO_INCOME_RATIO"]) - (data["AMT_ANNUITY"])), np.maximum(
        ((((-1.0) - (data["NEW_ANNUITY_TO_INCOME_RATIO"])))), ((((data["ORGANIZATION_TYPE_Mobile"]) * (((data["FONDKAPREMONT_MODE_org_spec_account"]) - (data["AMT_ANNUITY"])))))))))
    v["i208"] = 0.099849 * np.tanh(np.where(data["AMT_ANNUITY"] > 0, data["NAME_TYPE_SUITE_Children"], np.where(data["NAME_TYPE_SUITE_Children"] < 0, np.maximum(
        ((data["ORGANIZATION_TYPE_Realtor"])), ((data["OCCUPATION_TYPE_Secretaries"]))), ((data["NAME_TYPE_SUITE_Children"]) * ((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (0.318310)) / 2.0)) * 2.0))))))
    v["i209"] = 0.093200 * np.tanh(np.where(data["OCCUPATION_TYPE_Cleaning_staff"] < 0, (((2.0) < ((((((((((data["OCCUPATION_TYPE_Private_service_staff"]) + (1.570796)) / 2.0)) + (
        data["NEW_ANNUITY_TO_INCOME_RATIO"])) / 2.0)) + (data["NEW_ANNUITY_TO_INCOME_RATIO"])) / 2.0))) * 1.), ((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["NEW_ANNUITY_TO_INCOME_RATIO"]))))
    v["i210"] = 0.097000 * np.tanh((-1.0 * (((((((((((-1.0 * ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) > (data["OCCUPATION_TYPE_Private_service_staff"])) * 1.)) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))
                                               < (np.where((((0.54348003864288330)) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) < 0, data["OCCUPATION_TYPE_Private_service_staff"], (0.54348003864288330)))) * 1.)))))
    v["i211"] = 0.099991 * np.tanh(((np.maximum((((((((data["NAME_CONTRACT_TYPE_Cash_loans"]) + (-1.0))) < (data["AMT_ANNUITY"])) * 1.))), ((data["AMT_ANNUITY"])))) * (
        (((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, -1.0, data["NAME_CONTRACT_TYPE_Cash_loans"])) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.))))
    v["i212"] = 0.086070 * np.tanh(((((data["DAYS_REGISTRATION"]) * (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (np.tanh((np.where(
        data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, data["DAYS_REGISTRATION"], ((data["AMT_ANNUITY"]) + (((data["AMT_ANNUITY"]) + (-2.0)))))))))))) / 2.0))
    v["i213"] = 0.080000 * np.tanh(((((((((data["NAME_CONTRACT_TYPE_Revolving_loans"]) > (((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)) * (3.141593)))) * 1.)) + (
        data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) < (np.minimum((((((data["ORGANIZATION_TYPE_Housing"]) > (3.141593)) * 1.))), ((data["NAME_TYPE_SUITE_Other_A"]))))) * 1.))
    v["i214"] = 0.019990 * np.tanh((-1.0 * (((((((((((data["LIVE_REGION_NOT_WORK_REGION"]) + ((((data["LIVE_REGION_NOT_WORK_REGION"]) + (
        1.570796)) / 2.0))) / 2.0)) + (1.570796)) / 2.0)) > (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (2.0)))) * 1.)))))
    v["i215"] = 0.077040 * np.tanh(np.where((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (2.0)) * 1.) > 0, ((((((((data["AMT_ANNUITY"]) * 2.0)) * 2.0)) * 2.0))
                                                                                                         * 2.0), ((data["NAME_TYPE_SUITE_Other_A"]) * (((data["ORGANIZATION_TYPE_Housing"]) + (data["NEW_ANNUITY_TO_INCOME_RATIO"]))))))
    v["i216"] = 0.095410 * np.tanh((-1.0 * ((((np.maximum(((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) - (0.636620))) - (0.636620)))), ((np.maximum(
        ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["ORGANIZATION_TYPE_Mobile"])))), ((data["ORGANIZATION_TYPE_Trade__type_5"]))))))) * (data["NEW_DOC_IND_KURT"]))))))
    v["i217"] = 0.040000 * np.tanh(((((data["ORGANIZATION_TYPE_Mobile"]) * 2.0)) - (((((np.where(np.where(data["NAME_INCOME_TYPE_Maternity_leave"] > 0, data[
                                   "NEW_CREDIT_TO_INCOME_RATIO"], data["FLAG_EMAIL"]) < 0, data["ORGANIZATION_TYPE_Mobile"], data["AMT_ANNUITY"])) * (data["NEW_CREDIT_TO_INCOME_RATIO"]))) * 2.0))))
    v["i218"] = 0.099979 * np.tanh((-1.0 * ((np.where(((data["AMT_ANNUITY"]) * (data["NAME_TYPE_SUITE_Other_A"])) > 0, data["NAME_TYPE_SUITE_Other_A"], np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]
                                                                                                                                                                 < 0, (((data["FONDKAPREMONT_MODE_org_spec_account"]) > (((data["AMT_ANNUITY"]) * (data["NAME_TYPE_SUITE_Other_A"])))) * 1.), data["FONDKAPREMONT_MODE_org_spec_account"]))))))
    v["i219"] = 0.040997 * np.tanh(np.where(data["ORGANIZATION_TYPE_Industry__type_12"] > 0, ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0), (((-1.0 * (
        (((data["WALLSMATERIAL_MODE_Mixed"]) * (np.where(((data["AMT_ANNUITY"]) + (data["NEW_ANNUITY_TO_INCOME_RATIO"])) > 0, data["NEW_ANNUITY_TO_INCOME_RATIO"], data["AMT_ANNUITY"]))))))) / 2.0)))
    v["i220"] = 0.092382 * np.tanh((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (((3.0) / 2.0)))) < (np.where(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])
                                                                                                                * (data["NAME_INCOME_TYPE_Maternity_leave"])) < 0, 1.570796, data["NAME_INCOME_TYPE_Maternity_leave"]))) * 1.)) * 2.0))
    v["i221"] = 0.095900 * np.tanh(((np.where(((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) + (
        2.0)) < 0, -2.0, ((((((2.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) / 2.0)) < (0.636620)) * 1.))) / 2.0))
    v["i222"] = 0.037996 * np.tanh(((((((np.minimum(((0.0)), ((data["DAYS_BIRTH"])))) * (((np.minimum(((data["REG_CITY_NOT_WORK_CITY"])), ((
        data["DAYS_ID_PUBLISH"])))) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))) - ((((0.636620) < (data["DAYS_BIRTH"])) * 1.)))) / 2.0))
    v["i223"] = 0.099900 * np.tanh(np.minimum(((((data["HOUR_APPR_PROCESS_START"]) * (((((1.34566104412078857)) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.))))), ((
        (-1.0 * ((((((((1.570796) < (data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.)) < (((data["HOUR_APPR_PROCESS_START"]) - ((1.34566104412078857))))) * 1.))))))))
    v["i224"] = 0.099940 * np.tanh((-1.0 * ((np.maximum(((data["PREV_NAME_YIELD_GROUP_low_action_MEAN"])), ((((np.where(data["PREV_NAME_YIELD_GROUP_low_action_MEAN"]
                                                                                                                        < -99998, ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (-1.0)), -2.0)) * ((((-1.0 * ((-2.0)))) - (data["PREV_NAME_GOODS_CATEGORY_Furniture_MEAN"])))))))))))
    v["i225"] = 0.096000 * np.tanh(((np.minimum(((np.tanh((np.tanh((data["HOUR_APPR_PROCESS_START"])))))), (((((((data["APPROVED_RATE_DOWN_PAYMENT_MAX"]) > (
        data["APPROVED_RATE_DOWN_PAYMENT_MEAN"])) * 1.)) * (data["NEW_DOC_IND_KURT"])))))) * (np.maximum(((data["APPROVED_RATE_DOWN_PAYMENT_MEAN"])), ((data["NEW_DOC_IND_KURT"]))))))
    v["i226"] = 0.040300 * np.tanh((((((((((((((np.maximum(((data["PREV_AMT_DOWN_PAYMENT_MAX"])), ((((data["PREV_AMT_DOWN_PAYMENT_MEAN"]) * 2.0))))) > (1.570796)) * 1.)) - (
        data["PREV_RATE_DOWN_PAYMENT_MEAN"]))) + (data["PREV_AMT_DOWN_PAYMENT_MAX"])) / 2.0)) - (data["PREV_AMT_DOWN_PAYMENT_MEAN"]))) + (data["PREV_AMT_DOWN_PAYMENT_MAX"])) / 2.0))
    v["i227"] = 0.076520 * np.tanh(((np.maximum(((((((((((3.141593) + (np.where(data["PREV_RATE_DOWN_PAYMENT_MEAN"] < 0, -2.0, data["PREV_RATE_DOWN_PAYMENT_MEAN"]))) / 2.0)) < (
        data["APPROVED_RATE_DOWN_PAYMENT_MAX"])) * 1.)) * 2.0))), (((((data["APPROVED_AMT_DOWN_PAYMENT_MEAN"]) > ((4.20197486877441406))) * 1.))))) * 2.0))
    v["i228"] = 0.099992 * np.tanh((-1.0 * (((((data["PREV_RATE_DOWN_PAYMENT_MEAN"]) > (((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]) + (np.tanh(
        (np.maximum((((((data["PREV_RATE_DOWN_PAYMENT_MEAN"]) + (1.570796)) / 2.0))), ((((1.0) - (data["APPROVED_AMT_DOWN_PAYMENT_MAX"]))))))))))) * 1.)))))
    v["i229"] = 0.061230 * np.tanh(np.tanh((((((((((3.0) / 2.0)) + (data["NAME_INCOME_TYPE_Student"])) / 2.0)) < (np.where(data["PREV_AMT_DOWN_PAYMENT_MEAN"] > 0, data[
                                   "PREV_AMT_DOWN_PAYMENT_MEAN"], (((((data["APPROVED_AMT_DOWN_PAYMENT_MEAN"]) + (data["PREV_RATE_DOWN_PAYMENT_MEAN"])) / 2.0)) * (3.0))))) * 1.))))
    v["i230"] = 0.099601 * np.tanh((-1.0 * (((((((-1.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) > (np.tanh((np.where(
        data["APPROVED_AMT_DOWN_PAYMENT_MEAN"] < -99998, ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (np.tanh((-2.0))))) * 2.0), 1.570796))))) * 1.)))))
    v["i231"] = 0.099999 * np.tanh(((0.318310) - (np.maximum((((((((data["APPROVED_AMT_GOODS_PRICE_MAX"]) > (data["NEW_CREDIT_TO_INCOME_RATIO"])) * 1.)) * (0.318310)))), ((
        ((np.tanh((((data["PREV_NAME_YIELD_GROUP_low_normal_MEAN"]) * 2.0)))) + (((data["PREV_CHANNEL_TYPE_Channel_of_corporate_sales_MEAN"]) / 2.0)))))))))
    v["i232"] = 0.068050 * np.tanh((-1.0 * (((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > ((1.75574588775634766))) * 1.))
                                              * (((((data["AMT_ANNUITY"]) + (data["AMT_ANNUITY"]))) - ((1.75574588775634766)))))))))
    v["i233"] = 0.041997 * np.tanh((((np.where((((((data["PREV_AMT_DOWN_PAYMENT_MEAN"]) / 2.0)) > (((data["APPROVED_RATE_DOWN_PAYMENT_MEAN"]) * 2.0))) * 1.)
                                               > 0, ((((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]) * 2.0)) * 2.0), (-1.0 * ((((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]) * 2.0)))))) > (0.636620)) * 1.))
    v["i234"] = 0.094000 * np.tanh(((np.where(data["PREV_AMT_DOWN_PAYMENT_MAX"] > 0, np.minimum(((((((((1.94404411315917969)) < (data["PREV_AMT_DOWN_PAYMENT_MEAN"])) * 1.)) * (
        (-1.0 * ((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]))))))), ((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]))), (((1.570796) < (data["PREV_RATE_DOWN_PAYMENT_MEAN"])) * 1.))) * 2.0))
    v["i235"] = 0.0 * np.tanh((((((np.maximum(((data["PREV_AMT_DOWN_PAYMENT_MAX"])), ((data["PREV_RATE_DOWN_PAYMENT_MAX"])))) - ((1.97025465965270996))))
                                > (np.where(((data["PREV_RATE_DOWN_PAYMENT_MAX"]) - (2.0)) > 0, ((data["PREV_AMT_DOWN_PAYMENT_MAX"]) * 2.0), ((1.0) / 2.0)))) * 1.))
    v["i236"] = 0.018970 * np.tanh((((1.0) < (((np.maximum(((data["PREV_AMT_DOWN_PAYMENT_MAX"])), ((data["PREV_RATE_DOWN_PAYMENT_MAX"])))) - (
        ((np.minimum(((2.0)), ((data["APPROVED_AMT_DOWN_PAYMENT_MAX"])))) + (((0.636620) * 2.0))))))) * 1.))
    v["i237"] = 0.085574 * np.tanh(((((((np.maximum((((((((((data["APPROVED_AMT_DOWN_PAYMENT_MEAN"]) / 2.0)) > (np.where(data["APPROVED_AMT_DOWN_PAYMENT_MEAN"] < 0, data[
                                   "NAME_INCOME_TYPE_Maternity_leave"], 2.0))) * 1.)) * 2.0))), ((data["NAME_INCOME_TYPE_Maternity_leave"])))) * 2.0)) - (data["NAME_INCOME_TYPE_Maternity_leave"]))) * 2.0))
    v["i238"] = 0.095500 * np.tanh((((data["APPROVED_RATE_DOWN_PAYMENT_MEAN"]) > (np.maximum(((np.where(data["APPROVED_AMT_DOWN_PAYMENT_MAX"] > 0, data[
                                   "PREV_RATE_DOWN_PAYMENT_MEAN"], 1.570796))), ((np.where(data["APPROVED_RATE_DOWN_PAYMENT_MEAN"] > 0, 0.636620, data["PREV_RATE_DOWN_PAYMENT_MEAN"])))))) * 1.))
    v["i239"] = 0.090190 * np.tanh((((-1.0 * ((np.maximum(((((((((data["REFUSED_DAYS_DECISION_MAX"]) / 2.0)) / 2.0)) / 2.0))), ((np.where(data["REFUSED_DAYS_DECISION_MAX"]
                                                                                                                                          > 0, data["REFUSED_APP_CREDIT_PERC_MAX"], ((data["REFUSED_DAYS_DECISION_MAX"]) + ((-1.0 * ((data["REFUSED_APP_CREDIT_PERC_MAX"]))))))))))))) * 2.0))
    v["i240"] = 0.099500 * np.tanh(((np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] < 0, (((data["PREV_AMT_DOWN_PAYMENT_MEAN"]) > (0.318310)) * 1.), ((
        (((data["AMT_ANNUITY"]) + (((((((-2.0) + (data["PREV_AMT_DOWN_PAYMENT_MAX"])) / 2.0)) + (data["NEW_ANNUITY_TO_INCOME_RATIO"])) / 2.0))) / 2.0)) / 2.0))) / 2.0))
    v["i241"] = 0.097649 * np.tanh(np.where(data["APPROVED_AMT_GOODS_PRICE_MAX"] > 0, (((((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]) < ((((np.tanh((data["APPROVED_AMT_GOODS_PRICE_MAX"]))) + (
        ((-1.0) - (((data["NAME_INCOME_TYPE_Maternity_leave"]) / 2.0))))) / 2.0))) * 1.)) - (0.318310)), data["NAME_INCOME_TYPE_Maternity_leave"]))
    v["i242"] = 0.080299 * np.tanh(np.minimum((((-1.0 * (((((np.tanh((data["PREV_HOUR_APPR_PROCESS_START_MAX"]))) + ((((data["PREV_CHANNEL_TYPE_Channel_of_corporate_sales_MEAN"]) + (((data["PREV_PRODUCT_COMBINATION_Cash_Street__low_MEAN"]) + (
        data["PREV_NAME_PAYMENT_TYPE_Cash_through_the_bank_MEAN"])))) / 2.0))) / 2.0)))))), (((((((data["PREV_NAME_PAYMENT_TYPE_Cash_through_the_bank_MEAN"]) < (data["NAME_INCOME_TYPE_Maternity_leave"])) * 1.)) / 2.0)))))
    v["i243"] = 0.098999 * np.tanh(((((((data["REG_REGION_NOT_LIVE_REGION"]) * ((((data["HOUR_APPR_PROCESS_START"]) < (np.minimum(
        ((((np.tanh((-2.0))) / 2.0))), ((np.tanh(((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (-2.0)) / 2.0)))))))) * 1.)))) * 2.0)) * 2.0))
    v["i244"] = 0.099999 * np.tanh(((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < -99998, data["NEW_CREDIT_TO_ANNUITY_RATIO"], (((np.minimum(((np.tanh((np.tanh(
        (data["FLAG_WORK_PHONE"])))))), ((((data["ORGANIZATION_TYPE_Industry__type_2"]) * 2.0))))) > ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (0.636620)) / 2.0))) * 1.))) * 2.0))
    v["i245"] = 0.018790 * np.tanh((((((data["NAME_INCOME_TYPE_Maternity_leave"]) + (np.where(data["DAYS_BIRTH"] > 0, data["NEW_CREDIT_TO_INCOME_RATIO"],
                                                                                              np.where(data["AMT_CREDIT"] > 0, data["NEW_CREDIT_TO_INCOME_RATIO"], (-1.0 * ((data["NAME_INCOME_TYPE_Working"]))))))) / 2.0)) / 2.0))
    v["i246"] = 0.099560 * np.tanh(np.where(((((data["NEW_CREDIT_TO_INCOME_RATIO"]) * ((((data["NAME_FAMILY_STATUS_Widow"]) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)))) * 2.0) > 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"],
                                            np.where(data["NEW_CREDIT_TO_INCOME_RATIO"] > 0, (((data["NAME_INCOME_TYPE_Maternity_leave"]) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.), data["NAME_INCOME_TYPE_Maternity_leave"])))
    v["i247"] = 0.099499 * np.tanh(((data["ORGANIZATION_TYPE_Agriculture"]) * (np.where(((data["REG_CITY_NOT_WORK_CITY"]) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) > 0, (((13.75214290618896484)) * (
        np.where(data["REG_CITY_NOT_WORK_CITY"] > 0, (-1.0 * ((data["ORGANIZATION_TYPE_Agriculture"]))), data["ORGANIZATION_TYPE_Agriculture"]))), data["NEW_ANNUITY_TO_INCOME_RATIO"]))))
    v["i248"] = 0.099999 * np.tanh((-1.0 * ((np.maximum((((((data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"]) > (((((((0.636620) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) + (
        data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) * (0.636620)))) * 1.))), ((np.maximum(((data["CC_NAME_CONTRACT_STATUS_Sent_proposal_VAR"])), ((data["ORGANIZATION_TYPE_Industry__type_2"]))))))))))
    v["i249"] = 0.099981 * np.tanh(np.where(data["CC_NAME_CONTRACT_STATUS_Active_MEAN"] > 0, np.where(data["CC_NAME_CONTRACT_STATUS_Active_SUM"] < 0, data["CC_AMT_BALANCE_MEAN"], (-1.0 * (
        (((((data["CC_AMT_BALANCE_MEAN"]) * 2.0)) * 2.0))))), (-1.0 * (((((data["CC_NAME_CONTRACT_STATUS_Refused_MEAN"]) < (((data["CC_AMT_BALANCE_MEAN"]) * 2.0))) * 1.))))))
    v["i250"] = 0.091701 * np.tanh(((data["CC_AMT_PAYMENT_CURRENT_VAR"]) - (np.where(np.where(((data["CC_AMT_PAYMENT_CURRENT_VAR"]) - (data["CC_AMT_PAYMENT_CURRENT_MAX"])) < 0, data[
                                   "CC_AMT_PAYMENT_CURRENT_MAX"], 2.0) > 0, data["CC_AMT_PAYMENT_CURRENT_MEAN"], (((((5.0)) + (data["CC_AMT_PAYMENT_CURRENT_VAR"]))) * 2.0)))))
    v["i251"] = 0.099670 * np.tanh((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)) * (
        (((data["OCCUPATION_TYPE_Sales_staff"]) > (((np.where(2.0 > 0, data["AMT_ANNUITY"], data["NEW_CREDIT_TO_ANNUITY_RATIO"])) + (np.tanh((1.0)))))) * 1.))))
    v["i252"] = 0.077998 * np.tanh(((((((np.where(data["NAME_CONTRACT_TYPE_Revolving_loans"] > 0, data["POS_NAME_CONTRACT_STATUS_Amortized_debt_MEAN"], np.where(
        (((data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"]) > ((5.0))) * 1.) > 0, (6.0), ((((12.61040878295898438)) < (data["POS_SK_DPD_MAX"])) * 1.)))) * 2.0)) * 2.0)) * 2.0))
    v["i253"] = 0.089490 * np.tanh((-1.0 * ((np.where(np.where((((data["ACTIVE_DAYS_CREDIT_VAR"]) > (data["CC_AMT_PAYMENT_CURRENT_MEAN"])) * 1.) > 0, 0.318310, data[
                                   "NAME_CONTRACT_TYPE_Cash_loans"]) < 0, ((data["OCCUPATION_TYPE_Managers"]) * (data["CC_AMT_PAYMENT_CURRENT_MEAN"])), (((data["CC_AMT_PAYMENT_CURRENT_MEAN"]) > (0.318310)) * 1.))))))
    v["i254"] = 0.099500 * np.tanh(np.where(data["BURO_CREDIT_TYPE_Mortgage_MEAN"] > 0, -2.0, np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"] > 0, data["NEW_DOC_IND_KURT"], np.where(
        data["BURO_CREDIT_TYPE_Car_loan_MEAN"] > 0, -2.0, (-1.0 * ((np.where(data["BURO_CREDIT_TYPE_Another_type_of_loan_MEAN"] < -99998, data["NEW_DOC_IND_KURT"], data["BURO_CREDIT_TYPE_Another_type_of_loan_MEAN"]))))))))
    v["i255"] = 0.099920 * np.tanh(np.where(data["BURO_CREDIT_TYPE_Loan_for_business_development_MEAN"] < -99998, 0.318310, (-1.0 * ((np.where(data["BURO_CREDIT_ACTIVE_Closed_MEAN"] < 0, data["BURO_CREDIT_TYPE_Mortgage_MEAN"],
                                                                                                                                               np.maximum(((0.0)), ((((np.where(data["BURO_CREDIT_TYPE_Mortgage_MEAN"] > 0, data["BURO_CREDIT_ACTIVE_Closed_MEAN"], data["BURO_CREDIT_TYPE_Loan_for_business_development_MEAN"])) * 2.0))))))))))
    v["i256"] = 0.099145 * np.tanh((((np.maximum(((((((((data["OWN_CAR_AGE"]) / 2.0)) / 2.0)) - (np.where(((1.0) - (data["DAYS_REGISTRATION"])) > 0, data[
                                   "OWN_CAR_AGE"], data["DAYS_REGISTRATION"]))))), ((data["NAME_INCOME_TYPE_Maternity_leave"])))) < (data["NEW_CAR_TO_EMPLOY_RATIO"])) * 1.))
    v["i257"] = 0.098100 * np.tanh(np.minimum((((((((data["NEW_SOURCES_PROD"]) > ((((-1.0 * ((data["OCCUPATION_TYPE_Managers"])))) * 2.0))) * 1.)) * 2.0))), ((
        np.where(data["NEW_SOURCES_PROD"] < 0, data["OCCUPATION_TYPE_Managers"], (((((-1.0 * ((data["OCCUPATION_TYPE_Managers"])))) * 2.0)) * 2.0))))))
    v["i258"] = 0.097465 * np.tanh((((((-1.0 * (((((data["EXT_SOURCE_1"]) > (data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.))))) * (
        data["EXT_SOURCE_1"]))) * (np.minimum((((((data["EXT_SOURCE_1"]) > (1.0)) * 1.))), ((data["EXT_SOURCE_1"]))))))
    v["i259"] = 0.097799 * np.tanh(((np.minimum(((np.where(data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"] < 0, np.maximum(((0.318310)), ((data["ACTIVE_DAYS_CREDIT_VAR"]))), np.where(
        data["OWN_CAR_AGE"] > 0, data["NEW_CAR_TO_BIRTH_RATIO"], data["NEW_ANNUITY_TO_INCOME_RATIO"])))), ((np.maximum(((0.318310)), ((data["ACTIVE_DAYS_CREDIT_VAR"]))))))) / 2.0))
    v["i260"] = 0.098379 * np.tanh(np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < 0, np.where(data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"] < 0, np.where(data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"] < -99998, ((
        (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) < (data["ACTIVE_DAYS_CREDIT_VAR"])) * 1.), data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]), ((-2.0) * (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]))), data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]))
    v["i261"] = 0.097180 * np.tanh(((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * ((-1.0 * ((((((((data["NEW_SCORES_STD"]) < ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (
        data["NEW_CAR_TO_EMPLOY_RATIO"])) / 2.0))) * 1.)) + ((((data["NEW_SCORES_STD"]) < (np.minimum(((-1.0)), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))) * 1.))) / 2.0)))))))
    v["i262"] = 0.085897 * np.tanh(((((((1.47757565975189209)) + (np.maximum(((0.318310)), ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (np.minimum(
        ((data["AMT_INCOME_TOTAL"])), (((0.55474889278411865)))))))))))) < (np.maximum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), ((((data["AMT_INCOME_TOTAL"]) * 2.0)))))) * 1.))
    v["i263"] = 0.095189 * np.tanh((((data["CC_NAME_CONTRACT_STATUS_Refused_MAX"]) > ((-1.0 * (((((data["ACTIVE_DAYS_CREDIT_VAR"]) > ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (
        ((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (0.318310)) * 1.)) + (np.maximum(((data["AMT_ANNUITY"])), ((data["ACTIVE_DAYS_CREDIT_VAR"]))))) / 2.0))) / 2.0))) * 1.)))))) * 1.))
    v["i264"] = 0.090129 * np.tanh(np.minimum((((((np.where(((0.318310) - (data["CC_AMT_PAYMENT_CURRENT_MEAN"])) < 0, data["CC_AMT_PAYMENT_CURRENT_MAX"], (((-1.0) < (
        data["ACTIVE_DAYS_CREDIT_VAR"])) * 1.))) < (data["CC_AMT_PAYMENT_CURRENT_MAX"])) * 1.))), ((((data["CC_AMT_PAYMENT_CURRENT_MAX"]) - (data["CC_AMT_PAYMENT_CURRENT_MEAN"]))))))
    v["i265"] = 0.056959 * np.tanh(np.where((((((((-1.0) > (data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.)) - ((1.16799139976501465)))) + (data["ACTIVE_DAYS_CREDIT_VAR"])) > 0, ((
        data["NEW_ANNUITY_TO_INCOME_RATIO"]) * 2.0), np.where(data["NAME_EDUCATION_TYPE_Academic_degree"] < 0, data["NAME_EDUCATION_TYPE_Academic_degree"], data["ACTIVE_DAYS_CREDIT_VAR"])))
    v["i266"] = 0.080598 * np.tanh(((((((data["AMT_ANNUITY"]) * (((((2.62576174736022949)) < (((data["ACTIVE_DAYS_CREDIT_VAR"]) * 2.0))) * 1.))))
                                      * ((4.19378137588500977)))) * (np.minimum(((3.0)), ((((data["ACTIVE_DAYS_CREDIT_VAR"]) * 2.0)))))))
    v["i267"] = 0.081050 * np.tanh(((np.tanh(((((data["ACTIVE_DAYS_CREDIT_VAR"]) > (data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.)))) * (np.where(data["WALLSMATERIAL_MODE_Block"] > 0,
                                                                                                                                               (-1.0 * ((data["WALLSMATERIAL_MODE_Block"]))), np.where(data["ACTIVE_DAYS_CREDIT_VAR"] > 0, (-1.0 * ((data["WALLSMATERIAL_MODE_Block"]))), data["ACTIVE_DAYS_CREDIT_VAR"])))))
    v["i268"] = 0.099500 * np.tanh((((((data["NAME_INCOME_TYPE_Maternity_leave"]) > (np.minimum(((((3.141593) - (data["NEW_SCORES_STD"])))), ((
        ((data["NEW_SCORES_STD"]) + (1.570796))))))) * 1.)) + ((-1.0 * (((((data["ACTIVE_DAYS_CREDIT_VAR"]) > (3.0)) * 1.)))))))
    v["i269"] = 0.100000 * np.tanh((((((((np.maximum(((data["ORGANIZATION_TYPE_Mobile"])), ((((((-2.0) + (data["DAYS_REGISTRATION"]))) + (
        np.tanh((1.570796)))))))) * 2.0)) + ((((data["DAYS_REGISTRATION"]) < (-2.0)) * 1.))) / 2.0)) * 2.0))
    v["i270"] = 0.018900 * np.tanh(((data["AMT_ANNUITY"]) * (np.tanh((np.tanh((np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, data["NEW_CREDIT_TO_ANNUITY_RATIO"],
                                                                                        np.tanh((np.where(-1.0 < -99998, data["AMT_ANNUITY"], ((data["OCCUPATION_TYPE_Laborers"]) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))))))))))
    v["i271"] = 0.099700 * np.tanh(np.where(((((1.89746308326721191)) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.) > 0, data["ACTIVE_DAYS_CREDIT_VAR"], ((
        ((1.89746308326721191)) < (((((data["AMT_ANNUITY"]) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) * (((data["AMT_ANNUITY"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))) * 1.)))
    v["i272"] = 0.099600 * np.tanh(np.where(((data["NEW_DOC_IND_KURT"]) + (3.0)) < 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (
        np.maximum(((np.where(data["NEW_DOC_IND_KURT"] < 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], ((data["AMT_ANNUITY"]) * 2.0)))), ((2.0))))) * 1.)))
    v["i273"] = 0.099500 * np.tanh(np.where((((((data["AMT_CREDIT"]) < ((((data["AMT_CREDIT"]) < (data["NAME_CONTRACT_TYPE_Revolving_loans"])) * 1.))) * 1.)) + (
        data["ACTIVE_DAYS_CREDIT_VAR"])) < 0, (((((data["AMT_CREDIT"]) > (3.141593)) * 1.)) * 2.0), np.tanh((data["NAME_CONTRACT_TYPE_Revolving_loans"]))))
    v["i274"] = 0.099951 * np.tanh(np.where(((data["NEW_CREDIT_TO_INCOME_RATIO"]) - (1.570796)) < 0, np.where(np.where(data["NEW_CREDIT_TO_INCOME_RATIO"] < 0, data["CLOSED_DAYS_CREDIT_VAR"], data[
                                   "EXT_SOURCE_3"]) > 0, data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"], data["NAME_INCOME_TYPE_Maternity_leave"]), ((data["CLOSED_DAYS_CREDIT_VAR"]) - (data["EXT_SOURCE_3"]))))
    v["i275"] = 0.075990 * np.tanh((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)) > (np.where((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (
        (((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < ((2.0))) * 1.)) / 2.0))) / 2.0) < 0, data["NAME_CONTRACT_TYPE_Revolving_loans"], 3.141593))) * 1.))
    v["i276"] = 0.078515 * np.tanh((-1.0 * (((((((((3.141593) < (((3.141593) * (data["AMT_ANNUITY"])))) * 1.)) * (
        (((data["EXT_SOURCE_3"]) + (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["AMT_CREDIT"])))) / 2.0)))) * (data["AMT_CREDIT"]))))))
    v["i277"] = 0.099698 * np.tanh((((0.318310) < (np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"] < -99998, data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"], ((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) * (
        np.where(data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"] > 0, data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"], np.where(data["AMT_CREDIT"] > 0, data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"], data["AMT_CREDIT"]))))))) * 1.))
    v["i278"] = 0.090481 * np.tanh(np.where(data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"] > 0, np.maximum((((-1.0 * (((((data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"]) > (
        (-1.0 * (((((((data["AMT_CREDIT"]) / 2.0)) > (0.636620)) * 1.)))))) * 1.)))))), ((data["CLOSED_DAYS_CREDIT_VAR"]))), data["NAME_INCOME_TYPE_Maternity_leave"]))
    v["i279"] = 0.098040 * np.tanh(((((((((data["EXT_SOURCE_3"]) * ((((1.570796) < (data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"])) * 1.)))) - (
        (((data["EXT_SOURCE_3"]) > (1.570796)) * 1.)))) - ((((data["EXT_SOURCE_3"]) > (1.570796)) * 1.)))) * 2.0))
    v["i280"] = 0.092470 * np.tanh((-1.0 * (((((np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"] < 0, (-1.0 * ((np.where(data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"] < 0, data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"],
                                                                                                                       np.where(data["EXT_SOURCE_3"] < 0, data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"], data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]))))), data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"])) < (-1.0)) * 1.)))))
    v["i281"] = 0.099961 * np.tanh(np.where(data["BURO_DAYS_CREDIT_ENDDATE_MEAN"] > 0, 0.318310, np.maximum((((((np.tanh((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]))) + (
        data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])) / 2.0))), ((np.maximum(((data["AMT_REQ_CREDIT_BUREAU_DAY"])), (((((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]) > (data["AMT_REQ_CREDIT_BUREAU_DAY"])) * 1.)))))))))
    v["i282"] = 0.042521 * np.tanh(np.tanh(((((-1.0 * ((np.tanh((((((0.636620) - ((-1.0 * ((data["EXT_SOURCE_3"])))))) * (
        np.where(data["AMT_CREDIT"] > 0, data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"], data["NEW_CAR_TO_EMPLOY_RATIO"]))))))))) / 2.0))))
    v["i283"] = 0.099848 * np.tanh((((-1.0 * ((data["FLAG_OWN_CAR"])))) * ((((data["BURO_DAYS_CREDIT_VAR"]) > ((((data["NEW_CAR_TO_BIRTH_RATIO"]) < (
        np.where(data["NEW_CAR_TO_BIRTH_RATIO"] < 0, data["ACTIVE_DAYS_CREDIT_VAR"], data["BURO_DAYS_CREDIT_VAR"]))) * 1.))) * 1.))))
    v["i284"] = 0.054018 * np.tanh(((((((np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, data["NEW_CAR_TO_BIRTH_RATIO"], data["OWN_CAR_AGE"])) < (0.636620)) * 1.)) < (
        np.where(data["BURO_DAYS_CREDIT_VAR"] < -99998, np.maximum(((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) / 2.0))), ((data["OWN_CAR_AGE"]))), 0.636620))) * 1.))
    v["i285"] = 0.033002 * np.tanh((-1.0 * (((((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, (2.93577146530151367), (3.75477886199951172))) < (
        np.where(((data["AMT_CREDIT"]) - ((2.93577146530151367))) < 0, ((data["AMT_CREDIT"]) + (data["AMT_ANNUITY"])), data["AMT_CREDIT"]))) * 1.)))))
    v["i286"] = 0.099998 * np.tanh(((((((data["OCCUPATION_TYPE_Laborers"]) + (np.minimum(((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] > 0, 2.0, 0.318310))),
                                                                                         ((((-1.0) - (data["ACTIVE_DAYS_CREDIT_VAR"]))))))) / 2.0)) > (((2.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) * 1.))
    v["i287"] = 0.099101 * np.tanh(((data["DAYS_ID_PUBLISH"]) * (((((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] > 0, (-1.0 * (((((0.636620) + (
        (-1.0 * ((data["NEW_ANNUITY_TO_INCOME_RATIO"]))))) / 2.0)))), 0.636620)) * ((-1.0 * ((data["NEW_ANNUITY_TO_INCOME_RATIO"])))))) / 2.0))))
    v["i288"] = 0.000299 * np.tanh(((((np.minimum(((data["ORGANIZATION_TYPE_Trade__type_4"])), ((np.where((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (data["ORGANIZATION_TYPE_Trade__type_4"])) * 1.)
                                                                                                          > 0, data["NAME_CONTRACT_TYPE_Cash_loans"], ((((data["ORGANIZATION_TYPE_Trade__type_4"]) * 2.0)) * (data["NAME_CONTRACT_TYPE_Revolving_loans"]))))))) * 2.0)) * 2.0))
    v["i289"] = 0.038261 * np.tanh(np.where((((data["ORGANIZATION_TYPE_Trade__type_4"]) > (np.minimum(((np.minimum(((data["AMT_ANNUITY"])), ((((data["AMT_ANNUITY"]) * (
        data["NAME_CONTRACT_TYPE_Revolving_loans"]))))))), ((data["ORGANIZATION_TYPE_Trade__type_4"]))))) * 1.) > 0, data["ORGANIZATION_TYPE_Trade__type_4"], ((-2.0) * 2.0)))
    v["i290"] = 0.096999 * np.tanh(np.minimum(((np.minimum((((((((-1.0 * ((np.tanh((((data["AMT_ANNUITY"]) + (0.636620)))))))) * 2.0)) * (data["NAME_CONTRACT_TYPE_Revolving_loans"])))),
                                                           ((data["ORGANIZATION_TYPE_Trade__type_4"]))))), ((((data["ORGANIZATION_TYPE_Trade__type_4"]) * (data["NAME_CONTRACT_TYPE_Revolving_loans"]))))))
    v["i291"] = 0.099975 * np.tanh((((((((data["ACTIVE_DAYS_CREDIT_VAR"]) + (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (data["CLOSED_MONTHS_BALANCE_MAX_MAX"]))))) > (
        np.maximum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), ((np.maximum((((((data["AMT_ANNUITY"]) + (1.0)) / 2.0))), ((0.318310)))))))) * 1.)) * 2.0))
    v["i292"] = 0.094888 * np.tanh((-1.0 * ((np.where((((data["ACTIVE_DAYS_CREDIT_VAR"]) < (data["CLOSED_MONTHS_BALANCE_MIN_MIN"])) * 1.) > 0, data["BURO_MONTHS_BALANCE_MAX_MAX"], ((
        (np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, data["NAME_INCOME_TYPE_Student"], data["BURO_MONTHS_BALANCE_MAX_MAX"])) > ((((data["ACTIVE_DAYS_CREDIT_VAR"]) > (0.636620)) * 1.))) * 1.))))))
    v["i293"] = 0.099000 * np.tanh(((3.0) * ((((data["EXT_SOURCE_3"]) < (np.where(data["EXT_SOURCE_3"] < -99998, data["ACTIVE_AMT_ANNUITY_MAX"], np.where(
        data["EXT_SOURCE_3"] > 0, data["BURO_AMT_ANNUITY_MEAN"], np.where(data["ACTIVE_AMT_ANNUITY_MAX"] > 0, data["EXT_SOURCE_3"], -2.0))))) * 1.))))
    v["i294"] = 0.010870 * np.tanh(np.tanh(((((data["AMT_ANNUITY"]) < (np.minimum(((np.where((((0.318310) + (
        data["AMT_ANNUITY"])) / 2.0) < 0, data["BURO_AMT_ANNUITY_MAX"], data["EXT_SOURCE_3"]))), ((0.318310))))) * 1.))))
    v["i295"] = 0.005211 * np.tanh(((np.where((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < (3.141593)) * 1.) > 0, ((np.where(((data["ORGANIZATION_TYPE_Trade__type_4"]) * (
        data["NAME_INCOME_TYPE_Student"])) > 0, ((data["NAME_INCOME_TYPE_Maternity_leave"]) * 2.0), data["EXT_SOURCE_3"])) * 2.0), data["EXT_SOURCE_3"])) * 2.0))
    v["i296"] = 0.096877 * np.tanh(np.where(((data["AMT_ANNUITY"]) + (data["LIVE_REGION_NOT_WORK_REGION"])) > 0, ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (
        data["ORGANIZATION_TYPE_Industry__type_12"])), (((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > ((((2.0)) + (data["LIVE_REGION_NOT_WORK_REGION"])))) * 1.)) * 2.0)) * 2.0)))
    v["i297"] = 0.099999 * np.tanh((((np.where(np.where(data["EXT_SOURCE_3"] < -99998, data["PREV_AMT_DOWN_PAYMENT_MAX"], data["EXT_SOURCE_3"])
                                               < -99998, -2.0, 0.318310)) + (((np.where(data["EXT_SOURCE_3"] < 0, -1.0, data["PREV_AMT_DOWN_PAYMENT_MAX"])) / 2.0))) / 2.0))
    v["i298"] = 0.021000 * np.tanh(((np.tanh((np.tanh(((-1.0 * ((np.where(data["APPROVED_RATE_DOWN_PAYMENT_MAX"] < 0, np.where(data["APPROVED_RATE_DOWN_PAYMENT_MEAN"] < 0, ((
        data["APPROVED_RATE_DOWN_PAYMENT_MEAN"]) / 2.0), data["APPROVED_RATE_DOWN_PAYMENT_MEAN"]), ((data["PREV_AMT_DOWN_PAYMENT_MEAN"]) * 2.0)))))))))) / 2.0))
    v["i299"] = 0.002602 * np.tanh(np.minimum(((((((-1.0 * ((np.tanh((data["APPROVED_AMT_DOWN_PAYMENT_MEAN"])))))) + (np.minimum((((((np.tanh((data["PREV_AMT_DOWN_PAYMENT_MEAN"]))) < (
        data["APPROVED_AMT_DOWN_PAYMENT_MEAN"])) * 1.))), (((-1.0 * ((data["BURO_CREDIT_TYPE_Another_type_of_loan_MEAN"])))))))) / 2.0))), (((-1.0 * ((data["BURO_CREDIT_TYPE_Another_type_of_loan_MEAN"])))))))
    v["i300"] = 0.099860 * np.tanh(np.where(data["PREV_RATE_DOWN_PAYMENT_MAX"] < -99998, 0.318310, ((((data["APPROVED_RATE_DOWN_PAYMENT_MEAN"]) * (
        ((((((((2.37128782272338867)) + (data["APPROVED_AMT_DOWN_PAYMENT_MEAN"]))) + (data["PREV_AMT_DOWN_PAYMENT_MAX"]))) < (data["APPROVED_RATE_DOWN_PAYMENT_MAX"])) * 1.)))) * 2.0)))
    v["i301"] = 0.000150 * np.tanh((-1.0 * ((((np.maximum((((((data["BURO_CREDIT_TYPE_Loan_for_business_development_MEAN"]) > (((((-1.0) + ((((data["BURO_CNT_CREDIT_PROLONG_SUM"]) < (
        (-1.0 * ((data["BURO_MONTHS_BALANCE_SIZE_SUM"]))))) * 1.)))) * (data["BURO_CREDIT_TYPE_Car_loan_MEAN"])))) * 1.))), ((data["BURO_CREDIT_TYPE_Mortgage_MEAN"])))) * 2.0)))))
    v["i302"] = 0.099600 * np.tanh((((-1.0 * (((((0.318310) < (np.where(data["BURO_CREDIT_TYPE_Car_loan_MEAN"] > 0, data["BURO_MONTHS_BALANCE_SIZE_SUM"], ((
        ((((data["BURO_MONTHS_BALANCE_SIZE_SUM"]) > (((data["BURO_CREDIT_TYPE_Consumer_credit_MEAN"]) / 2.0))) * 1.)) < (((data["BURO_MONTHS_BALANCE_SIZE_SUM"]) / 2.0))) * 1.)))) * 1.))))) * 2.0))
    v["i303"] = 0.070501 * np.tanh(np.where(data["CLOSED_AMT_ANNUITY_MEAN"] > 0, (((((((data["ACTIVE_MONTHS_BALANCE_MAX_MAX"]) * (data["ACTIVE_AMT_ANNUITY_MAX"]))) < (data["BURO_AMT_ANNUITY_MAX"])) * 1.)) + (
        ((data["ACTIVE_AMT_ANNUITY_MAX"]) * (data["BURO_AMT_ANNUITY_MAX"])))), np.maximum(((data["NAME_INCOME_TYPE_Maternity_leave"])), ((data["ACTIVE_MONTHS_BALANCE_MAX_MAX"])))))
    v["i304"] = 0.079600 * np.tanh((((-1.0 * ((np.maximum(((np.maximum(((data["BURO_CREDIT_TYPE_Loan_for_business_development_MEAN"])), ((data["BURO_CREDIT_TYPE_Mortgage_MEAN"]))))), ((
        (((data["BURO_CREDIT_TYPE_Loan_for_business_development_MEAN"]) > (((data["BURO_CREDIT_TYPE_Car_loan_MEAN"]) * (((np.maximum(((data["NAME_INCOME_TYPE_Student"])), ((data["BURO_CREDIT_TYPE_Another_type_of_loan_MEAN"])))) * 2.0))))) * 1.)))))))) * 2.0))
    v["i305"] = 0.080957 * np.tanh(np.where((((1.570796) < (data["BURO_CREDIT_TYPE_Mortgage_MEAN"])) * 1.) > 0, data["BURO_AMT_ANNUITY_MAX"], np.where(data["BURO_AMT_ANNUITY_MAX"] < -99998, ((
        ((((data["BURO_CREDIT_ACTIVE_Closed_MEAN"]) < (data["BURO_CREDIT_TYPE_Mortgage_MEAN"])) * 1.)) < (data["BURO_CREDIT_TYPE_Mortgage_MEAN"])) * 1.), (-1.0 * ((data["BURO_CREDIT_TYPE_Mortgage_MEAN"]))))))
    v["i306"] = 0.098040 * np.tanh(np.minimum(((((((-1.0 * ((data["BURO_CREDIT_TYPE_Consumer_credit_MEAN"])))) > (data["BURO_CREDIT_TYPE_Credit_card_MEAN"])) * 1.))), ((
        np.where(data["BURO_CREDIT_TYPE_Consumer_credit_MEAN"] > 0, ((((((-1.0 * ((data["BURO_CREDIT_TYPE_Credit_card_MEAN"])))) > (0.318310)) * 1.)) - (0.318310)), 0.318310)))))
    v["i307"] = 0.097210 * np.tanh(np.tanh((((((((1.0) + (data["CLOSED_DAYS_CREDIT_VAR"]))) + (data["NEW_SOURCES_PROD"]))) * (
        (((np.tanh((np.tanh((-1.0))))) < ((((data["NEW_SOURCES_PROD"]) + (data["AMT_REQ_CREDIT_BUREAU_DAY"])) / 2.0))) * 1.))))))
    v["i308"] = 0.034689 * np.tanh(np.maximum(((data["AMT_REQ_CREDIT_BUREAU_DAY"])), (((((0.636620) < (np.where(np.where(data["BURO_DAYS_CREDIT_ENDDATE_MEAN"] < 0, data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"], data[
                                   "NEW_SOURCES_PROD"]) < 0, np.where(data["NEW_SOURCES_PROD"] < -99998, data["BURO_DAYS_CREDIT_ENDDATE_MEAN"], data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]), data["NEW_SOURCES_PROD"]))) * 1.)))))
    v["i309"] = 0.098117 * np.tanh(((data["AMT_REQ_CREDIT_BUREAU_MON"]) * (np.where(data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"] < -99998, np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"] < -99998,
                                                                                                                                              np.where(data["AMT_REQ_CREDIT_BUREAU_MON"] < -99998, np.maximum(((0.0)), ((data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"]))), 3.141593), 2.0), data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"]))))
    v["i310"] = 0.099900 * np.tanh((((-1.0 * (((((np.where(np.maximum(((data["AMT_REQ_CREDIT_BUREAU_WEEK"])), ((data["AMT_REQ_CREDIT_BUREAU_QRT"]))) > 0, 1.570796, ((
        1.570796) * 2.0))) < (np.where(data["AMT_REQ_CREDIT_BUREAU_YEAR"] > 0, data["AMT_REQ_CREDIT_BUREAU_YEAR"], data["AMT_REQ_CREDIT_BUREAU_WEEK"]))) * 1.))))) * 2.0))
    v["i311"] = 0.091399 * np.tanh(((np.where((((data["NEW_SOURCES_PROD"]) + (1.0)) / 2.0) > 0, ((data["EXT_SOURCE_3"]) * 2.0), np.minimum(
        (((-1.0 * ((data["EXT_SOURCE_3"]))))), (((((data["EXT_SOURCE_3"]) > (data["NEW_SOURCES_PROD"])) * 1.)))))) / 2.0))
    v["i312"] = 0.080001 * np.tanh(np.where(data["NEW_SOURCES_PROD"] > 0, np.minimum(((data["EXT_SOURCE_3"])), ((data["AMT_REQ_CREDIT_BUREAU_YEAR"]))),
                                            ((0.0) - ((((0.318310) < (np.minimum(((data["AMT_REQ_CREDIT_BUREAU_QRT"])), ((data["EXT_SOURCE_3"]))))) * 1.)))))
    v["i313"] = 0.095050 * np.tanh(np.maximum(((data["AMT_REQ_CREDIT_BUREAU_DAY"])), ((np.where(data["NEW_SOURCES_PROD"] < 0, (((data["AMT_REQ_CREDIT_BUREAU_MON"]) > (np.where(
        data["NEW_SOURCES_PROD"] < -99998, data["AMT_REQ_CREDIT_BUREAU_MON"], ((data["NEW_SOURCES_PROD"]) + ((0.97451114654541016)))))) * 1.), data["AMT_REQ_CREDIT_BUREAU_MON"])))))
    v["i314"] = 0.095500 * np.tanh(((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]) * (np.minimum(((((2.0) - (data["AMT_REQ_CREDIT_BUREAU_MON"])))), ((
        ((((((data["AMT_REQ_CREDIT_BUREAU_YEAR"]) < (1.570796)) * 1.)) < (np.maximum(((data["AMT_REQ_CREDIT_BUREAU_QRT"])), ((data["CLOSED_DAYS_CREDIT_VAR"]))))) * 1.)))))))
    v["i315"] = 0.095929 * np.tanh(((((((((data["AMT_REQ_CREDIT_BUREAU_HOUR"]) < (data["AMT_REQ_CREDIT_BUREAU_MON"])) * 1.)) < (((((((data["EXT_SOURCE_3"]) < (
        data["AMT_REQ_CREDIT_BUREAU_WEEK"])) * 1.)) < (data["ACTIVE_DAYS_CREDIT_VAR"])) * 1.))) * 1.)) * (((((data["EXT_SOURCE_3"]) * 2.0)) + (data["AMT_REQ_CREDIT_BUREAU_YEAR"])))))
    v["i316"] = 0.070000 * np.tanh((((((-1.0 * ((np.where(data["AMT_REQ_CREDIT_BUREAU_DAY"] > 0, np.maximum(((data["AMT_REQ_CREDIT_BUREAU_WEEK"])), ((
        data["AMT_REQ_CREDIT_BUREAU_MON"]))), (((data["AMT_REQ_CREDIT_BUREAU_MON"]) > (((((11.36887073516845703)) + (2.0)) / 2.0))) * 1.)))))) * 2.0)) * 2.0))
    v["i317"] = 0.095040 * np.tanh(np.minimum(((((np.where(data["AMT_REQ_CREDIT_BUREAU_YEAR"] < 0, (-1.0 * ((np.where(data["AMT_REQ_CREDIT_BUREAU_MON"] < 0, 0.318310, data[
                                   "AMT_REQ_CREDIT_BUREAU_YEAR"])))), 0.318310)) / 2.0))), (((((6.07505083084106445)) + ((-1.0 * ((data["AMT_REQ_CREDIT_BUREAU_MON"])))))))))
    v["i318"] = 0.097585 * np.tanh((-1.0 * (((((((data["CLOSED_DAYS_CREDIT_VAR"]) / 2.0)) > ((((((0.318310) / 2.0)) < (np.where(data["CLOSED_MONTHS_BALANCE_MIN_MIN"] < -99998, data[
                                   "CLOSED_DAYS_CREDIT_VAR"], np.where(data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"] < -99998, data["CLOSED_DAYS_CREDIT_VAR"], data["CLOSED_MONTHS_BALANCE_MIN_MIN"])))) * 1.))) * 1.)))))
    v["i319"] = 0.099702 * np.tanh(((data["NEW_SCORES_STD"]) * ((((np.maximum(((0.318310)), ((data["EXT_SOURCE_3"])))) < (np.minimum(
        ((data["NEW_SCORES_STD"])), ((((((((data["EXT_SOURCE_3"]) + (data["NAME_INCOME_TYPE_Student"])) / 2.0)) > ((-1.0 * ((0.318310))))) * 1.)))))) * 1.))))
    v["i320"] = 0.062500 * np.tanh(np.where(data["ACTIVE_CNT_CREDIT_PROLONG_SUM"] > 0, (((((data["ACTIVE_MONTHS_BALANCE_SIZE_SUM"]) * (data["ACTIVE_CNT_CREDIT_PROLONG_SUM"]))) + (
        (5.0))) / 2.0), (((data["ACTIVE_CNT_CREDIT_PROLONG_SUM"]) > (np.maximum(((data["CLOSED_AMT_CREDIT_SUM_DEBT_MEAN"])), (((-1.0 * ((data["BURO_DAYS_CREDIT_VAR"])))))))) * 1.)))
    v["i321"] = 0.095000 * np.tanh(np.where(data["ACTIVE_CNT_CREDIT_PROLONG_SUM"] < 0, (((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]) > (((((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]) > (
        (0.79923409223556519))) * 1.)) + (1.0)) / 2.0))) * 1.), ((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) + (((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]) + (data["ACTIVE_MONTHS_BALANCE_SIZE_SUM"]))))))
    v["i322"] = 0.099003 * np.tanh((((((((((9.0)) + ((9.0)))) * (data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]))) * ((((-1.0) < (
        ((((data["ACTIVE_DAYS_CREDIT_VAR"]) + (data["ACTIVE_CREDIT_DAY_OVERDUE_MEAN"]))) + (data["ACTIVE_MONTHS_BALANCE_SIZE_SUM"])))) * 1.)))) * 2.0))
    v["i323"] = 0.099460 * np.tanh((((data["NAME_INCOME_TYPE_Maternity_leave"]) + ((((np.tanh((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) < (np.where(
        data["BURO_AMT_ANNUITY_MAX"] > 0, ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)) - (0.636620)), data["BURO_AMT_ANNUITY_MAX"]))) * 1.))) / 2.0))
    v["i324"] = 0.098000 * np.tanh((-1.0 * ((((np.where(data["BURO_STATUS_0_MEAN_MEAN"] < 0, (((-1.0) < (data["BURO_STATUS_0_MEAN_MEAN"])) * 1.), data["BURO_MONTHS_BALANCE_MAX_MAX"])) - (
        (((data["BURO_STATUS_0_MEAN_MEAN"]) < (np.where(data["CLOSED_MONTHS_BALANCE_MIN_MIN"] < 0, data["CLOSED_MONTHS_BALANCE_MIN_MIN"], data["BURO_STATUS_0_MEAN_MEAN"]))) * 1.)))))))
    v["i325"] = 0.099999 * np.tanh(((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)) * ((((data["CODE_GENDER"]) + (((((((data["CODE_GENDER"]) + (
        np.maximum(((data["CODE_GENDER"])), ((data["AMT_ANNUITY"]))))) / 2.0)) + (np.tanh((((data["AMT_ANNUITY"]) * 2.0))))) / 2.0))) / 2.0))))
    v["i326"] = 0.095525 * np.tanh((-1.0 * ((((np.maximum(((data["NAME_INCOME_TYPE_Working"])), ((data["NAME_INCOME_TYPE_Student"])))) * ((((((0.318310) + (np.maximum(
        ((data["DAYS_BIRTH"])), ((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]))))) / 2.0)) * ((((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]) + (data["DAYS_BIRTH"])) / 2.0)))))))))
    v["i327"] = 0.099698 * np.tanh(((np.minimum((((((3.0)) + (((data["ACTIVE_MONTHS_BALANCE_MIN_MIN"]) - (np.where(data["ACTIVE_MONTHS_BALANCE_MAX_MAX"] < 0, data["ACTIVE_MONTHS_BALANCE_MAX_MAX"], data[
                                   "NEW_ANNUITY_TO_INCOME_RATIO"]))))))), ((np.maximum(((data["ACTIVE_MONTHS_BALANCE_MAX_MAX"])), ((data["NAME_INCOME_TYPE_Maternity_leave"]))))))) - (data["NAME_INCOME_TYPE_Student"])))
    v["i328"] = 0.002697 * np.tanh(((np.maximum(((data["ORGANIZATION_TYPE_Military"])), ((((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) - (
        (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (-1.0)) * 1.))))))) * (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] > 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], -1.0))))
    v["i329"] = 0.059002 * np.tanh(((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * ((((1.570796) < (np.maximum(((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (
        1.570796))) / 2.0))), ((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] > 0, data["AMT_ANNUITY"], data["ORGANIZATION_TYPE_Realtor"])))))) * 1.))))
    v["i330"] = 0.088199 * np.tanh(((np.where(((((5.0)) < (data["AMT_CREDIT"])) * 1.) > 0, data["AMT_CREDIT"], (-1.0 * (
        (((((((4.84836912155151367)) / 2.0)) < (np.maximum(((data["AMT_ANNUITY"])), ((data["AMT_CREDIT"]))))) * 1.)))))) - (data["ORGANIZATION_TYPE_Trade__type_4"])))
    v["i331"] = 0.099984 * np.tanh((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["ORGANIZATION_TYPE_Military"]))) * (data["ORGANIZATION_TYPE_Military"]))) + (
        (((((data["AMT_ANNUITY"]) > (1.570796)) * 1.)) * (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))) / 2.0))
    v["i332"] = 0.099749 * np.tanh((((((data["NAME_FAMILY_STATUS_Separated"]) + (data["AMT_ANNUITY"])) / 2.0)) * ((-1.0 * ((((((((((((data["NAME_FAMILY_STATUS_Separated"]) + (
        data["AMT_CREDIT"])) / 2.0)) * (data["ORGANIZATION_TYPE_Realtor"]))) * (data["NAME_FAMILY_STATUS_Separated"]))) < (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])) * 1.)))))))
    v["i333"] = 0.098593 * np.tanh(np.where(((((-1.0 * ((((data["ORGANIZATION_TYPE_Industry__type_11"]) / 2.0))))) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) / 2.0) < 0, ((
        (((data["ORGANIZATION_TYPE_Industry__type_11"]) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)) - (((data["ORGANIZATION_TYPE_Industry__type_11"]) * (data["ORGANIZATION_TYPE_Realtor"])))), data["ORGANIZATION_TYPE_Realtor"]))
    v["i334"] = 0.000801 * np.tanh(((((data["ORGANIZATION_TYPE_Telecom"]) * (((data["ORGANIZATION_TYPE_Telecom"]) * (data["CODE_GENDER"]))))) - (
        ((((((((data["ORGANIZATION_TYPE_Telecom"]) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) / 2.0)) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) > (3.141593)) * 1.))))
    v["i335"] = 0.099815 * np.tanh((-1.0 * ((((((((2.0) + (3.141593)) / 2.0)) < (((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (
        np.minimum(((data["CODE_GENDER"])), ((((data["CODE_GENDER"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))))) * 2.0))) * 1.)))))
    v["i336"] = 0.090800 * np.tanh((((((np.where(((data["OCCUPATION_TYPE_Drivers"]) * (data["AMT_ANNUITY"])) > 0, data["OCCUPATION_TYPE_Drivers"], ((
        data["AMT_ANNUITY"]) * (data["NEW_ANNUITY_TO_INCOME_RATIO"])))) - (((data["OCCUPATION_TYPE_Drivers"]) * (data["AMT_ANNUITY"]))))) > (3.141593)) * 1.))
    v["i337"] = 0.074504 * np.tanh(((data["ORGANIZATION_TYPE_Trade__type_2"]) * (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (((data["ORGANIZATION_TYPE_Trade__type_2"]) + (
        ((data["ORGANIZATION_TYPE_Trade__type_2"]) * (np.where(data["AMT_ANNUITY"] > 0, ((data["ORGANIZATION_TYPE_Trade__type_2"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"])), data["AMT_ANNUITY"]))))))))))
    v["i338"] = 0.097991 * np.tanh(np.where(data["ORGANIZATION_TYPE_Business_Entity_Type_1"] < 0, (((((2.0) < (((np.tanh((data["EXT_SOURCE_3"]))) - (
        data["NEW_ANNUITY_TO_INCOME_RATIO"])))) * 1.)) * 2.0), ((((data["EXT_SOURCE_3"]) - (data["NEW_ANNUITY_TO_INCOME_RATIO"]))) - (data["NEW_ANNUITY_TO_INCOME_RATIO"]))))
    v["i339"] = 0.095998 * np.tanh(((((np.where(data["CLOSED_MONTHS_BALANCE_MIN_MIN"] < -99998, ((np.maximum(((data["NAME_INCOME_TYPE_Maternity_leave"])), ((data["ORGANIZATION_TYPE_Industry__type_12"])))) - (
        data["NAME_EDUCATION_TYPE_Academic_degree"])), data["ORGANIZATION_TYPE_Business_Entity_Type_1"])) - (data["NAME_INCOME_TYPE_Student"]))) - (data["ORGANIZATION_TYPE_Industry__type_12"])))
    v["i340"] = 0.026998 * np.tanh(((((((data["CLOSED_AMT_ANNUITY_MEAN"]) < (data["AMT_ANNUITY"])) * 1.)) < (((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, np.tanh(
        (data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((data["CLOSED_AMT_ANNUITY_MEAN"]) - (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))) / 2.0))) * 1.))
    v["i341"] = 0.098498 * np.tanh(np.where(np.maximum(((((data["NEW_SOURCES_PROD"]) + (((((data["NEW_SOURCES_PROD"]) / 2.0)) + (data["NEW_ANNUITY_TO_INCOME_RATIO"])))))), ((
        data["NEW_SOURCES_PROD"]))) < 0, data["ORGANIZATION_TYPE_Telecom"], ((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (data["NEW_ANNUITY_TO_INCOME_RATIO"]))))
    v["i342"] = 0.097800 * np.tanh(((((np.where(data["NEW_SCORES_STD"] > 0, ((np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] > 0, data["NEW_SCORES_STD"], -1.0)) / 2.0), np.where(
        data["NEW_ANNUITY_TO_INCOME_RATIO"] > 0, data["NEW_SCORES_STD"], data["AMT_INCOME_TOTAL"]))) * (data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]))) / 2.0))
    v["i343"] = 0.098500 * np.tanh(((((np.tanh((np.where(data["NEW_CREDIT_TO_INCOME_RATIO"] > 0, (-1.0 * ((data["NEW_SCORES_STD"]))), ((
        ((-1.0 * (((0.35045036673545837))))) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.))))) / 2.0)) - ((((3.0) < (data["NEW_CREDIT_TO_INCOME_RATIO"])) * 1.))))
    v["i344"] = 0.093999 * np.tanh((-1.0 * (((((((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] > 0, data["ORGANIZATION_TYPE_Government"], ((-1.0) / 2.0))) - (
        np.maximum(((data["AMT_ANNUITY"])), ((np.maximum(((data["ORGANIZATION_TYPE_Government"])), ((data["ORGANIZATION_TYPE_Industry__type_12"]))))))))) < (-2.0)) * 1.)))))
    v["i345"] = 0.099994 * np.tanh((((((-1.0 * ((data["EXT_SOURCE_2"])))) * 2.0)) - (np.where(data["EXT_SOURCE_2"] < 0, ((
        (((data["EXT_SOURCE_2"]) * 2.0)) < (data["NEW_CREDIT_TO_INCOME_RATIO"])) * 1.), (((-1.0) + (((data["EXT_SOURCE_2"]) * 2.0))) / 2.0)))))
    v["i346"] = 0.096999 * np.tanh(((((((((data["REGION_POPULATION_RELATIVE"]) / 2.0)) < (data["REGION_RATING_CLIENT"])) * 1.)) < (((data["AMT_CREDIT"]) * (
        (((((((data["REGION_POPULATION_RELATIVE"]) / 2.0)) / 2.0)) + ((((data["REGION_RATING_CLIENT"]) > (data["REGION_POPULATION_RELATIVE"])) * 1.))) / 2.0))))) * 1.))
    v["i347"] = 0.079000 * np.tanh(((np.minimum(((np.where(data["YEARS_BEGINEXPLUATATION_MODE"] > 0, np.where(data["YEARS_BEGINEXPLUATATION_MEDI"] > 0, data["LIVINGAREA_MODE"],
                                                                                                              np.minimum(((-2.0)), ((data["HOUSETYPE_MODE_block_of_flats"])))), (-1.0 * ((data["LIVINGAREA_MEDI"])))))), ((((data["HOUSETYPE_MODE_block_of_flats"]) / 2.0))))) / 2.0))
    v["i348"] = 0.100000 * np.tanh(((3.0) * ((-1.0 * (((((3.0) < (np.where(data["REGION_RATING_CLIENT_W_CITY"]
                                                                           > 0, (-1.0 * ((((((data["DAYS_BIRTH"]) * 2.0)) * 2.0)))), ((data["DAYS_BIRTH"]) * 2.0)))) * 1.)))))))
    v["i349"] = 0.096042 * np.tanh((-1.0 * ((np.where(((data["WALLSMATERIAL_MODE_Wooden"]) + (((data["NAME_INCOME_TYPE_Student"]) + (data["NONLIVINGAREA_MODE"])))) > 0, np.where(
        data["WALLSMATERIAL_MODE_Wooden"] < 0, 0.318310, ((data["WALLSMATERIAL_MODE_Wooden"]) + (data["YEARS_BUILD_AVG"]))), data["WALLSMATERIAL_MODE_Wooden"])))))
    v["i350"] = 0.062002 * np.tanh((((((2.0) < (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["AMT_ANNUITY"])))) * 1.)) * (
        ((((((((-1.0 * ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) * 2.0)) > (data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.)) + (data["NAME_FAMILY_STATUS_Civil_marriage"])))))
    v["i351"] = 0.095559 * np.tanh(((np.where((((0.318310) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) / 2.0) > 0, 2.0, data["ORGANIZATION_TYPE_Advertising"])) * (
        np.minimum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), ((((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (3.141593)) / 2.0)) > (3.0)) * 1.)))))))
    v["i352"] = 0.095491 * np.tanh((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) > (1.570796)) * 1.)) - (
        (((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) * (1.570796))) > (3.0)) * 1.))))
    v["i353"] = 0.082139 * np.tanh(np.where((((3.0)) + (data["NEW_DOC_IND_KURT"])) > 0, ((data["NAME_CONTRACT_TYPE_Cash_loans"]) * (np.where(
        data["NEW_DOC_IND_KURT"] < 0, data["ACTIVE_MONTHS_BALANCE_MAX_MAX"], ((data["NAME_CONTRACT_TYPE_Revolving_loans"]) + (data["NEW_DOC_IND_KURT"]))))), 0.318310))
    v["i354"] = 0.084900 * np.tanh(((np.where(data["CLOSED_AMT_ANNUITY_MAX"] < 0, (((data["ORGANIZATION_TYPE_Advertising"]) + ((((8.0)) * (
        (((((((-1.0 * ((data["ORGANIZATION_TYPE_Advertising"])))) + (data["ACTIVE_MONTHS_BALANCE_MIN_MIN"])) / 2.0)) > (0.636620)) * 1.))))) / 2.0), data["ACTIVE_MONTHS_BALANCE_MAX_MAX"])) * 2.0))
    v["i355"] = 0.099010 * np.tanh(np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] < 0, ((data["ORGANIZATION_TYPE_Agriculture"]) * (((-1.0) - (data["AMT_ANNUITY"])))), np.minimum(
        ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["NAME_FAMILY_STATUS_Civil_marriage"])))), ((((data["ORGANIZATION_TYPE_Agriculture"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))))
    v["i356"] = 0.099990 * np.tanh(np.where(data["REFUSED_DAYS_DECISION_MEAN"] < -99998, (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (1.0)) * 1.), ((np.where((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (
        data["REFUSED_DAYS_DECISION_MEAN"])) / 2.0) < 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], data["NAME_INCOME_TYPE_Maternity_leave"])) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))
    v["i357"] = 0.098000 * np.tanh(((((((np.maximum(((data["REFUSED_AMT_GOODS_PRICE_MAX"])), ((np.minimum(((((data["REFUSED_HOUR_APPR_PROCESS_START_MEAN"]) * (
        ((-2.0) / 2.0))))), ((data["REFUSED_CNT_PAYMENT_SUM"]))))))) + (((-2.0) / 2.0))) / 2.0)) < (((data["REFUSED_CNT_PAYMENT_SUM"]) * 2.0))) * 1.))
    v["i358"] = 0.099497 * np.tanh((((((data["REFUSED_AMT_ANNUITY_MEAN"]) > (np.where(data["REFUSED_AMT_GOODS_PRICE_MEAN"] > 0, data["REFUSED_APP_CREDIT_PERC_MAX"], data[
                                   "REFUSED_AMT_GOODS_PRICE_MEAN"]))) * 1.)) * (((((((((data["REFUSED_AMT_ANNUITY_MAX"]) > (1.570796)) * 1.)) > (data["REFUSED_CNT_PAYMENT_MEAN"])) * 1.)) * (-2.0)))))
    v["i359"] = 0.065000 * np.tanh((((((np.maximum(((1.0)), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) > ((((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) < ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < (
        (((data["YEARS_BEGINEXPLUATATION_AVG"]) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.))) * 1.))) * 1.))) * 1.)) * (((data["YEARS_BEGINEXPLUATATION_AVG"]) * 2.0))))
    v["i360"] = 0.079021 * np.tanh((((((data["LANDAREA_AVG"]) < (((((((0.0)) * ((0.0)))) < (data["YEARS_BUILD_MODE"])) * 1.))) * 1.)) - (
        ((((((((data["LANDAREA_AVG"]) / 2.0)) > (data["LIVINGAPARTMENTS_MEDI"])) * 1.)) > (data["LIVINGAPARTMENTS_MEDI"])) * 1.))))
    v["i361"] = 0.066000 * np.tanh(((((data["NONLIVINGAPARTMENTS_AVG"]) * ((((-1.0) < (data["YEARS_BUILD_MODE"])) * 1.)))) * (
        (((np.tanh((np.minimum(((data["LIVINGAPARTMENTS_MEDI"])), ((data["YEARS_BUILD_MODE"])))))) < (((((data["LIVINGAPARTMENTS_MEDI"]) / 2.0)) / 2.0))) * 1.))))
    v["i362"] = 0.095000 * np.tanh(((((data["NEW_CREDIT_TO_INCOME_RATIO"]) * ((((0.318310) < ((((data["LIVINGAPARTMENTS_MEDI"]) + (
        ((((((data["LIVINGAPARTMENTS_MODE"]) > (((0.318310) / 2.0))) * 1.)) < (data["LIVINGAPARTMENTS_MEDI"])) * 1.))) / 2.0))) * 1.)))) * 2.0))
    v["i363"] = 0.078090 * np.tanh((((((((((np.where(((data["YEARS_BUILD_AVG"]) - (0.318310)) < 0, data["YEARS_BUILD_MODE"], ((data["LIVINGAPARTMENTS_MODE"]) + (
        data["LIVINGAPARTMENTS_MODE"])))) * (data["LIVINGAPARTMENTS_AVG"]))) < (data["NONLIVINGAPARTMENTS_MODE"])) * 1.)) * 2.0)) * 2.0))
    v["i364"] = 0.004800 * np.tanh((((-1.0 * (((((data["NONLIVINGAPARTMENTS_MODE"]) < (data["LIVINGAPARTMENTS_AVG"])) * 1.))))) + ((((2.0) < (((np.where(
        data["NONLIVINGAPARTMENTS_MODE"] < 0, data["LIVINGAPARTMENTS_MEDI"], data["NONLIVINGAPARTMENTS_MODE"])) + (np.tanh((data["LIVINGAPARTMENTS_AVG"])))))) * 1.))))
    v["i365"] = 0.028724 * np.tanh(((data["NONLIVINGAPARTMENTS_MODE"]) - (np.where(data["NONLIVINGAPARTMENTS_MODE"] < 0, data[
                                   "NONLIVINGAPARTMENTS_AVG"], (((((3.0) + (data["LIVINGAPARTMENTS_AVG"])) / 2.0)) - ((-1.0 * ((data["COMMONAREA_AVG"])))))))))
    v["i366"] = 0.045100 * np.tanh((((np.minimum(((((((data["COMMONAREA_AVG"]) + (data["NONLIVINGAPARTMENTS_MEDI"]))) * 2.0))), ((((np.minimum(
        (((((data["LIVINGAPARTMENTS_AVG"]) > (2.0)) * 1.))), ((data["NONLIVINGAPARTMENTS_MEDI"])))) / 2.0))))) > (data["LIVINGAPARTMENTS_AVG"])) * 1.))
    v["i367"] = 0.049702 * np.tanh(((np.minimum(((np.where((((data["COMMONAREA_MEDI"]) < (data["NONLIVINGAPARTMENTS_MEDI"])) * 1.) > 0, data[
                                   "NONLIVINGAPARTMENTS_MODE"], 1.570796))), (((((((((1.570796) + (data["LIVINGAPARTMENTS_MODE"]))) < (data["NONLIVINGAPARTMENTS_MEDI"])) * 1.)) * 2.0))))) * 2.0))
    v["i368"] = 0.095997 * np.tanh(((((np.minimum(((np.where(data["NONLIVINGAPARTMENTS_MEDI"] > 0, data["LIVINGAPARTMENTS_MEDI"], ((data["YEARS_BUILD_AVG"]) - (
        data["YEARS_BUILD_MEDI"]))))), ((((np.maximum(((data["NONLIVINGAPARTMENTS_MEDI"])), ((data["YEARS_BUILD_AVG"])))) - (data["YEARS_BUILD_MEDI"])))))) * 2.0)) * 2.0))
    v["i369"] = 0.002000 * np.tanh(((((data["YEARS_BUILD_AVG"]) - (data["YEARS_BUILD_MEDI"]))) * (((((data["LIVINGAPARTMENTS_MEDI"]) * (
        data["LIVINGAPARTMENTS_MODE"]))) * (np.maximum(((data["LIVINGAPARTMENTS_MODE"])), ((((0.636620) - (data["YEARS_BUILD_MODE"]))))))))))
    v["i370"] = 0.077003 * np.tanh((((((((((data["LIVINGAPARTMENTS_MODE"]) + (data["YEARS_BUILD_MEDI"]))) > (((data["LIVINGAPARTMENTS_MODE"]) + (
        data["YEARS_BUILD_MODE"])))) * 1.)) * (data["LIVINGAPARTMENTS_MODE"]))) * (np.maximum(((data["NONLIVINGAPARTMENTS_MODE"])), (((2.32663083076477051)))))))
    v["i371"] = 0.099997 * np.tanh((-1.0 * ((((((((((-1.0 * ((data["REFUSED_AMT_ANNUITY_MAX"])))) < (np.where(data["YEARS_BUILD_MODE"]
                                                                                                              > 0, (0.04093409702181816), np.where((0.0) > 0, data["REFUSED_AMT_ANNUITY_MAX"], (-1.0 * ((3.0))))))) * 1.)) * 2.0)) * 2.0)))))
    v["i372"] = 0.097999 * np.tanh((((data["REFUSED_AMT_APPLICATION_MAX"]) > (np.maximum(((((data["REFUSED_DAYS_DECISION_MAX"]) + (np.where(data["REFUSED_AMT_ANNUITY_MEAN"] > 0, np.tanh(
        (data["REFUSED_CNT_PAYMENT_SUM"])), data["REFUSED_AMT_APPLICATION_MAX"]))))), ((np.minimum(((data["REFUSED_AMT_ANNUITY_MEAN"])), ((np.tanh((data["REFUSED_CNT_PAYMENT_SUM"])))))))))) * 1.))
    v["i373"] = 0.099730 * np.tanh((-1.0 * ((((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (data["REFUSED_AMT_GOODS_PRICE_MAX"])) / 2.0)))
                                                   * 1.)) < (np.minimum((((((data["REFUSED_APP_CREDIT_PERC_MEAN"]) < (data["REFUSED_AMT_GOODS_PRICE_MAX"])) * 1.))), ((0.318310))))) * 1.)) * 2.0)))))
    v["i374"] = 0.072920 * np.tanh(((np.where(((((data["WALLSMATERIAL_MODE_Monolithic"]) * (data["NEW_CREDIT_TO_INCOME_RATIO"]))) - (data["NAME_INCOME_TYPE_Maternity_leave"])) > 0, np.where(
        data["NEW_CREDIT_TO_INCOME_RATIO"] < 0, data["WALLSMATERIAL_MODE_Monolithic"], 3.141593), data["ORGANIZATION_TYPE_Trade__type_7"])) * (((data["WALLSMATERIAL_MODE_Monolithic"]) * 2.0))))
    v["i375"] = 0.050000 * np.tanh(((((3.141593) * (((3.0) * (data["AMT_ANNUITY"]))))) * (
        (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (np.maximum(((((3.0) * (data["AMT_ANNUITY"])))), ((2.0))))) * 1.))))
    v["i376"] = 0.098411 * np.tanh(((np.minimum(((data["WALLSMATERIAL_MODE_Stone__brick"])), ((np.where(data["NEW_EXT_SOURCES_MEAN"] < 0, np.minimum(((data["AMT_ANNUITY"])), ((np.where(
        data["WALLSMATERIAL_MODE_Stone__brick"] < 0, data["NEW_CREDIT_TO_INCOME_RATIO"], data["ORGANIZATION_TYPE_School"])))), data["ORGANIZATION_TYPE_School"]))))) - (data["NEW_EXT_SOURCES_MEAN"])))
    v["i377"] = 0.100000 * np.tanh((-1.0 * ((np.where(data["NEW_SCORES_STD"] < 0, ((np.where((((data["NEW_EXT_SOURCES_MEAN"]) < ((-1.0 * ((data["DAYS_BIRTH"]))))) * 1.) > 0, data[
                                   "ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"], ((data["REGION_RATING_CLIENT"]) * 2.0))) - (data["NEW_EXT_SOURCES_MEAN"])), data["NEW_EXT_SOURCES_MEAN"])))))
    v["i378"] = 0.074051 * np.tanh((-1.0 * (((((((data["AMT_REQ_CREDIT_BUREAU_QRT"]) > ((-1.0 * ((data["DAYS_BIRTH"]))))) * 1.)) * (
        (((((data["AMT_REQ_CREDIT_BUREAU_QRT"]) > (np.tanh((np.tanh((np.tanh((data["EXT_SOURCE_3"])))))))) * 1.)) * (data["DAYS_BIRTH"]))))))))
    v["i379"] = 0.095001 * np.tanh((((data["EXT_SOURCE_3"]) > (((np.maximum(((np.maximum(((np.maximum(((((data["AMT_REQ_CREDIT_BUREAU_YEAR"]) * (data["AMT_REQ_CREDIT_BUREAU_YEAR"])))),
                                                                                                      ((data["EXT_SOURCE_3"]))))), ((data["AMT_REQ_CREDIT_BUREAU_QRT"]))))), ((np.tanh((3.141593)))))) + (data["BURO_DAYS_CREDIT_ENDDATE_MEAN"])))) * 1.))
    v["i380"] = 0.100000 * np.tanh(np.where(data["NEW_SOURCES_PROD"] < -99998, (-1.0 * ((((1.0) + (data["NEW_EXT_SOURCES_MEAN"]))))),
                                            ((((0.636620) + (((((((0.636620) + (data["NEW_SOURCES_PROD"]))) * 2.0)) * 2.0)))) * 2.0)))
    v["i381"] = 0.099600 * np.tanh(((((((((data["REGION_RATING_CLIENT_W_CITY"]) * (0.318310))) - (
        (((data["NEW_EXT_SOURCES_MEAN"]) < (((np.tanh((data["NEW_SCORES_STD"]))) + (-2.0)))) * 1.)))) * (data["NEW_EXT_SOURCES_MEAN"]))) * 2.0))
    v["i382"] = 0.099995 * np.tanh(np.where(((((0.61522376537322998)) + (data["NEW_EXT_SOURCES_MEAN"])) / 2.0) < 0, (((((0.318310) - (
        data["NEW_EXT_SOURCES_MEAN"]))) > ((-1.0 * ((-2.0))))) * 1.), (-1.0 * (((((data["NEW_SCORES_STD"]) + (data["NEW_EXT_SOURCES_MEAN"])) / 2.0))))))
    v["i383"] = 0.099400 * np.tanh(((np.minimum(((((((2.0) + (data["REGION_RATING_CLIENT"]))) + (
        data["DAYS_BIRTH"])))), (((((-1.0 * ((data["EXT_SOURCE_1"])))) + (-2.0)))))) * (data["DAYS_BIRTH"])))
    v["i384"] = 0.099699 * np.tanh(((((((((data["DAYS_REGISTRATION"]) > (0.0)) * 1.)) - (data["CODE_GENDER"]))) < (np.minimum(
        ((data["REGION_POPULATION_RELATIVE"])), ((np.minimum(((data["NEW_EXT_SOURCES_MEAN"])), ((data["WALLSMATERIAL_MODE_Panel"])))))))) * 1.))
    v["i385"] = 0.089970 * np.tanh((((((((np.maximum((((((np.minimum(((data["NEW_EXT_SOURCES_MEAN"])), ((data["NEW_SCORES_STD"])))) + (2.0)) / 2.0))),
                                                     ((((np.minimum(((1.570796)), ((data["NEW_SCORES_STD"])))) / 2.0))))) < (data["NEW_CAR_TO_BIRTH_RATIO"])) * 1.)) * 2.0)) * 2.0))
    v["i386"] = 0.099006 * np.tanh((((2.52555441856384277)) * (np.minimum(((np.maximum(((((((1.0) - (data["NEW_EXT_SOURCES_MEAN"]))) * 2.0))),
                                                                                       ((data["REFUSED_AMT_DOWN_PAYMENT_MAX"]))))), (((((0.636620) < (data["NEW_EXT_SOURCES_MEAN"])) * 1.)))))))
    v["i387"] = 0.099991 * np.tanh((-1.0 * ((((((np.tanh((np.where(-1.0 < 0, ((((-1.0 * ((data["REFUSED_AMT_DOWN_PAYMENT_MEAN"])))) < (
        ((data["FLAG_DOCUMENT_3"]) - (data["DAYS_BIRTH"])))) * 1.), data["REFUSED_AMT_DOWN_PAYMENT_MEAN"])))) * 2.0)) * (data["FLAG_DOCUMENT_3"]))))))
    v["i388"] = 0.099990 * np.tanh(np.where(data["NAME_FAMILY_STATUS_Single___not_married"] > 0, (-1.0 * ((((data["DAYS_BIRTH"]) + (np.minimum(((data["NAME_INCOME_TYPE_Student"])), ((
        data["REG_CITY_NOT_WORK_CITY"])))))))), np.maximum(((data["NAME_INCOME_TYPE_Student"])), (((-1.0 * ((((data["NEW_EXT_SOURCES_MEAN"]) + ((1.65381467342376709)))))))))))
    v["i389"] = 0.082100 * np.tanh(((data["NAME_EDUCATION_TYPE_Higher_education"]) * (((data["NEW_EXT_SOURCES_MEAN"]) * (
        (((((np.maximum(((((data["NEW_EXT_SOURCES_MEAN"]) * (data["HOUR_APPR_PROCESS_START"])))), ((data["HOUR_APPR_PROCESS_START"])))) > (1.570796)) * 1.)) + (((-1.0) / 2.0))))))))
    v["i390"] = 0.072495 * np.tanh(((data["AMT_REQ_CREDIT_BUREAU_DAY"]) * ((((data["NEW_EXT_SOURCES_MEAN"]) > (((np.tanh(((((((data["NEW_EXT_SOURCES_MEAN"]) > (
        data["AMT_REQ_CREDIT_BUREAU_DAY"])) * 1.)) + (np.minimum(((data["NEW_EXT_SOURCES_MEAN"])), ((0.318310)))))))) + (data["HOUR_APPR_PROCESS_START"])))) * 1.))))
    v["i391"] = 0.045818 * np.tanh(((np.maximum((((((data["NEW_DOC_IND_KURT"]) < (-2.0)) * 1.))), (((((data["NAME_EDUCATION_TYPE_Incomplete_higher"]) > (
        data["NAME_INCOME_TYPE_Student"])) * 1.))))) - ((((((data["OCCUPATION_TYPE_Cooking_staff"]) > (data["NEW_DOC_IND_KURT"])) * 1.)) / 2.0))))
    v["i392"] = 0.099760 * np.tanh(((data["NAME_FAMILY_STATUS_Civil_marriage"]) * (((np.maximum(((((np.maximum(((((data["NAME_FAMILY_STATUS_Civil_marriage"]) * (data["ORGANIZATION_TYPE_School"])))),
                                                                                                               (((-1.0 * ((data["DAYS_BIRTH"]))))))) - (data["NAME_FAMILY_STATUS_Civil_marriage"])))), ((data["DAYS_BIRTH"])))) * (data["ORGANIZATION_TYPE_School"])))))
    v["i393"] = 0.050400 * np.tanh(np.where(data["NEW_EXT_SOURCES_MEAN"] < -99998, data["NEW_EXT_SOURCES_MEAN"], np.where((((data["NEW_EXT_SOURCES_MEAN"]) + (data["OCCUPATION_TYPE_High_skill_tech_staff"])
                                                                                                                            ) / 2.0) > 0, (-1.0 * ((data["NAME_TYPE_SUITE_Group_of_people"]))), ((((((data["NEW_EXT_SOURCES_MEAN"]) + (2.0)) / 2.0)) < (data["OCCUPATION_TYPE_Core_staff"])) * 1.))))
    v["i394"] = 0.096996 * np.tanh((((((data["OCCUPATION_TYPE_High_skill_tech_staff"]) > (((data["NEW_EXT_SOURCES_MEAN"]) * (data["OCCUPATION_TYPE_High_skill_tech_staff"])))) * 1.)) * (
        np.where(((data["NEW_EXT_SOURCES_MEAN"]) * (data["REG_CITY_NOT_WORK_CITY"])) > 0, data["OCCUPATION_TYPE_High_skill_tech_staff"], data["WALLSMATERIAL_MODE_Panel"]))))
    v["i395"] = 0.075010 * np.tanh(((((((np.maximum((((((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]) > ((((np.minimum(((data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"])), ((
        data["NAME_INCOME_TYPE_Student"])))) + (((1.570796) - (data["NEW_EXT_SOURCES_MEAN"])))) / 2.0))) * 1.))), ((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"])))) * 2.0)) * 2.0)) * 2.0))
    v["i396"] = 0.100000 * np.tanh((((((((np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"])), ((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]) + (
        (-1.0 * ((data["ACTIVE_AMT_CREDIT_SUM_SUM"]))))))))) * 2.0)) > (np.where(data["ACTIVE_AMT_CREDIT_SUM_SUM"] < 0, 0.318310, 0.636620))) * 1.)) * 2.0))
    v["i397"] = 0.099999 * np.tanh((-1.0 * (((((((data["ACTIVE_AMT_CREDIT_SUM_MEAN"]) > (np.minimum(((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])), ((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]))))) * 1.)) * (
        (((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]) > (((np.minimum(((data["ACTIVE_AMT_CREDIT_SUM_SUM"])), ((data["ACTIVE_CNT_CREDIT_PROLONG_SUM"])))) + (data["ACTIVE_CNT_CREDIT_PROLONG_SUM"])))) * 1.)))))))
    v["i398"] = 0.099104 * np.tanh(((((np.where(data["NEW_EXT_SOURCES_MEAN"] < -99998, data["NEW_EXT_SOURCES_MEAN"], data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"])) - (np.where(np.minimum(
        ((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"])), (((((data["NEW_EXT_SOURCES_MEAN"]) > (data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"])) * 1.)))) > 0, data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"], data["ACTIVE_CREDIT_DAY_OVERDUE_MEAN"])))) * 2.0))
    v["i399"] = 0.095000 * np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"] < -99998, 0.0, ((data["ACTIVE_DAYS_CREDIT_MEAN"]) * (((np.where((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) > (
        data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"])) * 1.) > 0, ((0.318310) * 2.0), data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])) * (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]))))))
    v["i400"] = 0.100000 * np.tanh(np.where(data["ACTIVE_MONTHS_BALANCE_SIZE_SUM"] > 0, data["NEW_EXT_SOURCES_MEAN"], np.where(data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"] < 0, np.where(
        (((data["NEW_EXT_SOURCES_MEAN"]) > (1.570796)) * 1.) > 0, data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"], (((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]) > (0.636620)) * 1.)), data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"])))
    v["i401"] = 0.090000 * np.tanh(np.where(np.where(data["ACTIVE_CNT_CREDIT_PROLONG_SUM"] > 0, (-1.0 * ((data["NEW_EXT_SOURCES_MEAN"]))), data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]) > 0, ((
        data["ACTIVE_DAYS_CREDIT_MEAN"]) + (data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"])), (((1.570796) < (((data["NEW_EXT_SOURCES_MEAN"]) + (data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"])))) * 1.)))
    v["i402"] = 0.084600 * np.tanh((((((((np.where(data["ACTIVE_CNT_CREDIT_PROLONG_SUM"] > 0, data["ACTIVE_DAYS_CREDIT_MEAN"], data["ACTIVE_AMT_CREDIT_SUM_SUM"])) + (np.minimum(
        ((data["ACTIVE_CREDIT_DAY_OVERDUE_MEAN"])), ((data["ACTIVE_DAYS_CREDIT_MEAN"])))))) + (np.minimum(((data["ACTIVE_CNT_CREDIT_PROLONG_SUM"])), ((0.636620)))))) > (data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"])) * 1.))
    v["i403"] = 0.098501 * np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"] > 0, np.where(data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"] < 0, ((((((data["ACTIVE_DAYS_CREDIT_MEAN"]) < (data["ACTIVE_CREDIT_DAY_OVERDUE_MEAN"])) * 1.))
                                                                                                                                                   < ((-1.0 * ((data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"]))))) * 1.), data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]), ((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) - (data["ACTIVE_CREDIT_DAY_OVERDUE_MEAN"]))))
    v["i404"] = 0.099100 * np.tanh(np.where(data["NEW_EXT_SOURCES_MEAN"] < -99998, data["NEW_EXT_SOURCES_MEAN"], ((data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"]) * (
        (-1.0 * (((((((0.318310) * ((((((data["CLOSED_DAYS_CREDIT_VAR"]) / 2.0)) < (data["ACTIVE_DAYS_CREDIT_VAR"])) * 1.)))) < (data["ACTIVE_DAYS_CREDIT_VAR"])) * 1.))))))))
    v["i405"] = 0.098018 * np.tanh((-1.0 * ((np.where(np.where(np.where(data["BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"] < -99998, ((data["ACTIVE_DAYS_CREDIT_VAR"]) + (3.0)), data[
                                   "CLOSED_AMT_CREDIT_SUM_DEBT_MEAN"]) > 0, data["NEW_EXT_SOURCES_MEAN"], data["NAME_INCOME_TYPE_Student"]) > 0, (4.06813240051269531), data["NAME_INCOME_TYPE_Student"])))))
    v["i406"] = 0.097565 * np.tanh(np.where((((((data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]) * 2.0)) > (data["BURO_DAYS_CREDIT_VAR"])) * 1.) > 0, data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"], (-1.0 * (
        ((((((np.where(data["BURO_DAYS_CREDIT_VAR"] > 0, data["NAME_INCOME_TYPE_Student"], data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"])) < (data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"])) * 1.)) * 2.0))))))
    v["i407"] = 0.099599 * np.tanh(np.where(data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"] > 0, (((data["DAYS_BIRTH"]) + (((data["NEW_EXT_SOURCES_MEAN"]) * 2.0))) / 2.0), np.maximum(
        (((((data["BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"]) > ((0.04560829326510429))) * 1.))), (((((-1.0 * (((0.17060284316539764))))) * (data["DAYS_BIRTH"])))))))
    v["i408"] = 0.100000 * np.tanh(np.where(data["NEW_EXT_SOURCES_MEAN"] < -99998, data["NEW_EXT_SOURCES_MEAN"], (((0.0) > (((np.where(data["ACTIVE_MONTHS_BALANCE_MIN_MIN"] < 0, data[
                                   "NEW_EXT_SOURCES_MEAN"], np.where(data["NEW_EXT_SOURCES_MEAN"] < 0, data["ACTIVE_MONTHS_BALANCE_MAX_MAX"], data["YEARS_BUILD_AVG"]))) + ((2.71755051612854004))))) * 1.)))
    v["i409"] = 0.050000 * np.tanh(np.where(data["OCCUPATION_TYPE_Secretaries"] > 0, 3.0, np.where(data["NEW_EXT_SOURCES_MEAN"] > 0, data["POS_SK_DPD_MEAN"], (((np.where(
        data["POS_SK_DPD_MEAN"] > 0, 3.0, (((data["OCCUPATION_TYPE_Secretaries"]) + (data["NAME_TYPE_SUITE_Group_of_people"])) / 2.0))) < (data["POS_SK_DPD_MEAN"])) * 1.))))
    v["i410"] = 0.086952 * np.tanh(np.where(data["BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"] < -99998, np.maximum(((((data["NAME_INCOME_TYPE_Maternity_leave"]) * 2.0))), ((
        (((data["AMT_ANNUITY"]) > ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < (((data["HOUR_APPR_PROCESS_START"]) + ((2.70714354515075684))))) * 1.))) * 1.)))), data["BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"]))
    v["i411"] = 0.088340 * np.tanh(((((((((np.where(data["EXT_SOURCE_3"] < 0, 1.570796, 0.636620)) + (1.0)) / 2.0)) < (
        data["NEW_EXT_SOURCES_MEAN"])) * 1.)) * (((((data["EXT_SOURCE_3"]) - (1.0))) * (3.0)))))
    v["i412"] = 0.098500 * np.tanh((-1.0 * ((((((((((data["NEW_EXT_SOURCES_MEAN"]) + (data["NEW_SCORES_STD"])) / 2.0)) + (
        (((((data["NEW_EXT_SOURCES_MEAN"]) + (((data["NEW_SCORES_STD"]) + (data["NEW_EXT_SOURCES_MEAN"]))))) > (1.570796)) * 1.)))) > (1.570796)) * 1.)))))
    v["i413"] = 0.093700 * np.tanh((((((3.141593) < (np.where(data["AMT_ANNUITY"] > 0, data["AMT_ANNUITY"], ((np.maximum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), ((
        ((data["HOUR_APPR_PROCESS_START"]) * 2.0))))) * (np.maximum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), ((data["NEW_ANNUITY_TO_INCOME_RATIO"])))))))) * 1.)) * 2.0))
    v["i414"] = 0.099499 * np.tanh(np.where(data["BURO_DAYS_CREDIT_ENDDATE_MEAN"] < -99998, data["AMT_ANNUITY"], np.where((((np.tanh((0.636620))) + (np.where(
        data["NEW_EXT_SOURCES_MEAN"] < 0, np.tanh((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"])), data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) / 2.0) < 0, 3.0, data["NAME_INCOME_TYPE_Maternity_leave"])))
    v["i415"] = 0.082400 * np.tanh(np.where(data["NEW_DOC_IND_KURT"] > 0, 0.0, ((data["NEW_DOC_IND_KURT"]) + (np.maximum(((data["BURO_DAYS_CREDIT_VAR"])), ((
        ((data["NAME_CONTRACT_TYPE_Revolving_loans"]) - (np.where(data["BURO_DAYS_CREDIT_VAR"] > 0, data["BURO_DAYS_CREDIT_VAR"], data["CLOSED_AMT_CREDIT_SUM_MEAN"]))))))))))
    v["i416"] = 0.095004 * np.tanh((((((((data["CC_AMT_BALANCE_MEAN"]) > ((((2.03789639472961426)) - (np.where(
        data["AMT_ANNUITY"] < 0, (-1.0 * ((data["AMT_ANNUITY"]))), (((-1.0 * ((data["AMT_ANNUITY"])))) * 2.0)))))) * 1.)) * 2.0)) * 2.0))
    v["i417"] = 0.099799 * np.tanh(np.where((((data["NEW_EXT_SOURCES_MEAN"]) < (data["CC_AMT_PAYMENT_CURRENT_MEAN"])) * 1.) > 0, ((data["NEW_EXT_SOURCES_MEAN"]) - (
        data["AMT_ANNUITY"])), (((data["CC_AMT_PAYMENT_CURRENT_MAX"]) > (np.where(data["CC_AMT_PAYMENT_CURRENT_MEAN"] < 0, ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0), 1.0))) * 1.)))
    v["i418"] = 0.099600 * np.tanh((((np.where(((((((-2.0) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) / 2.0)) - (((data["NEW_EXT_SOURCES_MEAN"]) * 2.0)))
                                               < 0, np.tanh((data["CC_CNT_DRAWINGS_CURRENT_MEAN"])), data["NEW_CREDIT_TO_ANNUITY_RATIO"])) > (np.tanh((0.636620)))) * 1.))
    v["i419"] = 0.094997 * np.tanh((-1.0 * ((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + ((((0.318310) < ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (data["NEW_EXT_SOURCES_MEAN"])) / 2.0)))
                                                                                            * 1.))) / 2.0)) + (((((((0.318310) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) / 2.0)) < (data["ORGANIZATION_TYPE_Cleaning"])) * 1.))) / 2.0)))))
    v["i420"] = 0.080000 * np.tanh(((data["NEW_EXT_SOURCES_MEAN"]) * (((data["DAYS_BIRTH"]) * ((-1.0 * (
        ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > ((((np.tanh((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) > (((1.0) / 2.0))) * 1.))) * 1.)))))))))
    v["i421"] = 0.099805 * np.tanh((((np.where(data["DAYS_BIRTH"] < 0, ((np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] < 0, 2.0, data["NEW_EXT_SOURCES_MEAN"])) * (
        data["NEW_ANNUITY_TO_INCOME_RATIO"])), np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, 2.0, data["NEW_EXT_SOURCES_MEAN"]))) < (-2.0)) * 1.))
    v["i422"] = 0.099900 * np.tanh(np.where(data["NEW_EXT_SOURCES_MEAN"] < -99998, data["NEW_EXT_SOURCES_MEAN"], (((np.where(data["NEW_CREDIT_TO_INCOME_RATIO"]
                                                                                                                             < 0, ((data["NEW_EXT_SOURCES_MEAN"]) * (data["AMT_CREDIT"])), ((data["AMT_CREDIT"]) * (1.570796)))) < ((-1.0 * ((0.636620))))) * 1.)))
    v["i423"] = 0.098598 * np.tanh(((np.maximum(((data["NEW_EXT_SOURCES_MEAN"])), ((((data["AMT_ANNUITY"]) / 2.0))))) * (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (
        (((np.where(data["NEW_EXT_SOURCES_MEAN"] < 0, data["AMT_ANNUITY"], data["NEW_CREDIT_TO_ANNUITY_RATIO"])) > (((data["AMT_ANNUITY"]) / 2.0))) * 1.))))))
    v["i424"] = 0.099600 * np.tanh((-1.0 * ((((((data["NEW_EXT_SOURCES_MEAN"]) * 2.0)) * (np.maximum(((data["NAME_TYPE_SUITE_Group_of_people"])), ((
        ((((data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"]) + (((data["NEW_EXT_SOURCES_MEAN"]) / 2.0)))) + (((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)) * 2.0))))))))))))
    v["i425"] = 0.000051 * np.tanh((((((((data["ORGANIZATION_TYPE_Trade__type_3"]) + (((data["NEW_EXT_SOURCES_MEAN"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) / 2.0)) * (
        ((((data["ORGANIZATION_TYPE_Trade__type_3"]) * ((((-1.0 * ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) - (data["NEW_EXT_SOURCES_MEAN"]))))) * 2.0)))) / 2.0))
    v["i426"] = 0.090300 * np.tanh(((((((((0.318310) + ((((((((data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"]) + (data["NEW_ANNUITY_TO_INCOME_RATIO"])) / 2.0)) / 2.0)) / 2.0))
                                          ) / 2.0)) / 2.0)) < ((((data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"]) + (data["AMT_ANNUITY"])) / 2.0))) * 1.))
    v["i427"] = 0.025502 * np.tanh((((-1.0 * ((((((-1.0 * (((((data["CLOSED_MONTHS_BALANCE_MAX_MAX"]) < (data["CLOSED_MONTHS_BALANCE_MIN_MIN"])) * 1.))))) < (data["BURO_MONTHS_BALANCE_SIZE_MEAN"])) * 1.))))) + (
        np.where(data["BURO_STATUS_2_MEAN_MEAN"] > 0, data["BURO_STATUS_2_MEAN_MEAN"], (((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]) > (data["BURO_MONTHS_BALANCE_SIZE_MEAN"])) * 1.)))))
    v["i428"] = 0.084350 * np.tanh((-1.0 * ((((np.where(data["ACTIVE_DAYS_CREDIT_VAR"] > 0, data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"], 3.141593)) * (
        (((4.17438030242919922)) * ((((1.570796) < ((((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) + (np.tanh((3.141593)))) / 2.0))) * 1.)))))))))
    v["i429"] = 0.094973 * np.tanh((((((((3.0) < (data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.)) * (
        ((((-1.0) + (((((data["NEW_EXT_SOURCES_MEAN"]) + (3.0))) * (((data["NEW_EXT_SOURCES_MEAN"]) * 2.0)))))) * 2.0)))) * 2.0))
    v["i430"] = 0.100000 * np.tanh(np.where(data["CLOSED_MONTHS_BALANCE_MIN_MIN"] < 0, 0.0, ((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, data["NEW_ANNUITY_TO_INCOME_RATIO"],
                                                                                                       np.where(data["NEW_EXT_SOURCES_MEAN"] < 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], -2.0))) + (((data["NEW_EXT_SOURCES_MEAN"]) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))))
    v["i431"] = 0.099000 * np.tanh(((((((-2.0) + (data["NAME_INCOME_TYPE_Student"]))) + (((((np.maximum(((data["BURO_CREDIT_TYPE_Mortgage_MEAN"])), ((data["NEW_EXT_SOURCES_MEAN"])))) / 2.0))
                                                                                          * (data["NEW_EXT_SOURCES_MEAN"]))))) * (np.maximum(((data["BURO_CREDIT_TYPE_Mortgage_MEAN"])), ((data["NAME_TYPE_SUITE_Group_of_people"]))))))
    v["i432"] = 0.093000 * np.tanh(np.where(((data["NAME_EDUCATION_TYPE_Higher_education"]) + ((((2.72557568550109863)) - (data["AMT_ANNUITY"]))))
                                            > 0, (((((data["ORGANIZATION_TYPE_Industry__type_9"]) * (0.318310))) > ((((2.72557568550109863)) - (data["AMT_ANNUITY"])))) * 1.), -2.0))
    v["i433"] = 0.068043 * np.tanh((((((data["ORGANIZATION_TYPE_Trade__type_3"]) > ((((np.minimum(((data["REG_CITY_NOT_WORK_CITY"])), ((
        data["AMT_ANNUITY"])))) + ((((data["NEW_EXT_SOURCES_MEAN"]) < (1.570796)) * 1.))) / 2.0))) * 1.)) * ((-1.0 * ((data["REG_CITY_NOT_WORK_CITY"]))))))
    v["i434"] = 0.100000 * np.tanh(((np.where(((np.tanh((data["REG_REGION_NOT_LIVE_REGION"]))) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) < 0, np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] < 0, data[
                                   "NEW_CREDIT_TO_ANNUITY_RATIO"], data["WEEKDAY_APPR_PROCESS_START_WEDNESDAY"]), (-1.0 * ((((data["WEEKDAY_APPR_PROCESS_START_WEDNESDAY"]) * 2.0)))))) * (data["REG_REGION_NOT_LIVE_REGION"])))
    v["i435"] = 0.060100 * np.tanh(((np.where(data["AMT_ANNUITY"] < 0, data["NEW_ANNUITY_TO_INCOME_RATIO"], data["AMT_ANNUITY"])) * (
        np.tanh((((((((np.maximum(((data["AMT_ANNUITY"])), ((data["AMT_ANNUITY"])))) < (-1.0)) * 1.)) + (data["NEW_LIVE_IND_SUM"])) / 2.0))))))
    v["i436"] = 0.055846 * np.tanh((-1.0 * ((((((((data["AMT_ANNUITY"]) * (data["CLOSED_MONTHS_BALANCE_MIN_MIN"]))) + (data["AMT_ANNUITY"]))) * (
        (((data["CLOSED_MONTHS_BALANCE_MAX_MAX"]) > ((((((data["AMT_ANNUITY"]) * (data["CLOSED_MONTHS_BALANCE_MIN_MIN"]))) > (data["NEW_EXT_SOURCES_MEAN"])) * 1.))) * 1.)))))))
    v["i437"] = 0.086880 * np.tanh(((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (((np.where(data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"] < -99998, data["NEW_EXT_SOURCES_MEAN"], data[
                                   "NEW_ANNUITY_TO_INCOME_RATIO"])) * (np.where(data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"] > 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (2.0)) * 1.)))))))
    v["i438"] = 0.039851 * np.tanh(np.minimum((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["CLOSED_MONTHS_BALANCE_MAX_MAX"]))) > (-1.0)) * 1.))), ((((np.where(
        data["CLOSED_MONTHS_BALANCE_MAX_MAX"] < -99998, ((data["NAME_INCOME_TYPE_Student"]) / 2.0), data["CLOSED_MONTHS_BALANCE_MAX_MAX"])) * ((-1.0 * ((data["NEW_ANNUITY_TO_INCOME_RATIO"])))))))))
    v["i439"] = 0.086331 * np.tanh((-1.0 * (((((data["CLOSED_MONTHS_BALANCE_MIN_MIN"]) > (((2.0) - (np.where(data["CLOSED_MONTHS_BALANCE_MAX_MAX"] > 0, data[
                                   "NEW_ANNUITY_TO_INCOME_RATIO"], ((0.0) - (np.where(data["CLOSED_MONTHS_BALANCE_MIN_MIN"] > 0, data["NEW_EXT_SOURCES_MEAN"], data["CLOSED_MONTHS_BALANCE_MAX_MAX"])))))))) * 1.)))))
    v["i440"] = 0.099890 * np.tanh(np.where((((((np.tanh((0.318310))) / 2.0)) > (data["NEW_SCORES_STD"])) * 1.) > 0, ((data["NEW_EXT_SOURCES_MEAN"]) * (
        (((data["NAME_INCOME_TYPE_Maternity_leave"]) < (data["NEW_SCORES_STD"])) * 1.))), (((-1.0 * ((data["NEW_EXT_SOURCES_MEAN"])))) / 2.0)))
    v["i441"] = 0.099948 * np.tanh((((((((data["BURO_CREDIT_TYPE_Car_loan_MEAN"]) > ((((data["BURO_CREDIT_TYPE_Consumer_credit_MEAN"]) < (data["BURO_CREDIT_TYPE_Car_loan_MEAN"])) * 1.))) * 1.)) + (
        data["BURO_DAYS_CREDIT_MEAN"]))) * ((-1.0 * (((((data["BURO_CREDIT_TYPE_Car_loan_MEAN"]) > (((data["BURO_CREDIT_TYPE_Consumer_credit_MEAN"]) * (data["BURO_CREDIT_TYPE_Car_loan_MEAN"])))) * 1.)))))))
    v["i442"] = 0.080800 * np.tanh(np.where(data["CLOSED_AMT_CREDIT_SUM_DEBT_MEAN"] < 0, (-1.0 * (((((3.141593) < (((np.where((((data["CLOSED_AMT_CREDIT_SUM_DEBT_MEAN"]) < (data["CLOSED_AMT_CREDIT_SUM_SUM"]))
                                                                                                                               * 1.) > 0, data["CLOSED_MONTHS_BALANCE_SIZE_SUM"], data["CLOSED_DAYS_CREDIT_VAR"])) + (data["CLOSED_CREDIT_DAY_OVERDUE_MEAN"])))) * 1.)))), data["CLOSED_DAYS_CREDIT_VAR"]))
    v["i443"] = 0.067260 * np.tanh(np.where((((2.0) > (np.where(data["CLOSED_AMT_CREDIT_SUM_DEBT_MEAN"] < -99998, data["CLOSED_CNT_CREDIT_PROLONG_SUM"], data["CLOSED_AMT_CREDIT_SUM_MEAN"]))) * 1.)
                                            > 0, (((data["CLOSED_AMT_CREDIT_SUM_MEAN"]) > ((((data["CLOSED_AMT_CREDIT_SUM_SUM"]) > (np.tanh((data["CLOSED_CNT_CREDIT_PROLONG_SUM"])))) * 1.))) * 1.), -1.0))
    v["i444"] = 0.096199 * np.tanh((((-1.0 * (((((np.where(((data["CLOSED_CNT_CREDIT_PROLONG_SUM"]) + (data["CLOSED_AMT_CREDIT_SUM_MEAN"])) < 0, data["BURO_DAYS_CREDIT_VAR"],
                                                           np.maximum(((((data["CLOSED_MONTHS_BALANCE_SIZE_SUM"]) + (3.141593)))), ((data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]))))) > (3.141593)) * 1.))))) * 2.0))
    v["i445"] = 0.093960 * np.tanh((((np.maximum(((data["CLOSED_AMT_CREDIT_SUM_MEAN"])), ((data["CLOSED_AMT_CREDIT_SUM_DEBT_MEAN"])))) > (((((((-1.0 * ((np.maximum(((data["CLOSED_AMT_CREDIT_SUM_MEAN"])),
                                                                                                                                                                    ((data["CLOSED_AMT_CREDIT_SUM_DEBT_MEAN"]))))))) < ((((((data["CLOSED_AMT_CREDIT_SUM_DEBT_MEAN"]) + (data["CLOSED_AMT_CREDIT_SUM_SUM"])) / 2.0)) / 2.0))) * 1.)) / 2.0))) * 1.))
    v["i446"] = 0.099999 * np.tanh((((5.0)) * ((-1.0 * (((((((2.0) - (((np.where(data["CLOSED_AMT_CREDIT_SUM_SUM"] < 0, data[
                                   "CLOSED_AMT_CREDIT_SUM_DEBT_SUM"], 1.0)) - (data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]))))) < (data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"])) * 1.)))))))
    v["i447"] = 0.098930 * np.tanh(((np.where((((data["CLOSED_AMT_CREDIT_SUM_MEAN"]) < (np.where(data["BURO_DAYS_CREDIT_VAR"] < 0, data["CLOSED_AMT_CREDIT_SUM_DEBT_MEAN"], data[
                                   "CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN"]))) * 1.) > 0, (((data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]) > (0.636620)) * 1.), (((data["BURO_DAYS_CREDIT_VAR"]) > (2.0)) * 1.))) * 2.0))
    v["i448"] = 0.099380 * np.tanh((((((data["CLOSED_AMT_CREDIT_SUM_DEBT_MEAN"]) - ((((data["BURO_DAYS_CREDIT_VAR"]) < (data["CLOSED_CNT_CREDIT_PROLONG_SUM"])) * 1.)))) > (np.where(
        data["CLOSED_AMT_CREDIT_SUM_MEAN"] < 0, data["BURO_DAYS_CREDIT_VAR"], np.where(data["BURO_DAYS_CREDIT_VAR"] < 0, ((data["CLOSED_AMT_CREDIT_SUM_DEBT_MEAN"]) * 2.0), data["CLOSED_AMT_CREDIT_SUM_DEBT_MEAN"])))) * 1.))
    v["i449"] = 0.080300 * np.tanh(((-2.0) * ((((((((((data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]) > (-2.0)) * 1.)) + (data["CLOSED_AMT_CREDIT_SUM_SUM"])) / 2.0)) < (
        np.where(data["CLOSED_AMT_CREDIT_SUM_SUM"] < 0, data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"], data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]))) * 1.))))
    v["i450"] = 0.099520 * np.tanh(np.where(data["CLOSED_DAYS_CREDIT_MEAN"] < 0, (((np.minimum(((data["CLOSED_AMT_CREDIT_SUM_MEAN"])), ((data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"])))) > (
        data["CLOSED_AMT_CREDIT_SUM_SUM"])) * 1.), np.minimum((((((data["CLOSED_AMT_CREDIT_SUM_SUM"]) > ((((data["CLOSED_AMT_CREDIT_SUM_MEAN"]) > (0.318310)) * 1.))) * 1.))), ((data["CLOSED_AMT_CREDIT_SUM_SUM"])))))
    v["i451"] = 0.085997 * np.tanh((-1.0 * ((np.where(np.where(data["CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN"] < 0, data["CLOSED_MONTHS_BALANCE_SIZE_SUM"], data["CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN"])
                                                      < 0, (((data["CLOSED_AMT_CREDIT_SUM_MEAN"]) > (2.0)) * 1.), ((data["CLOSED_MONTHS_BALANCE_SIZE_SUM"]) * (data["CLOSED_AMT_CREDIT_SUM_DEBT_MEAN"])))))))
    v["i452"] = 0.047994 * np.tanh((((((((np.tanh((data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]))) + (data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]))) > (data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"])) * 1.)) * (
        np.where(data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"] < 0, data["CLOSED_AMT_CREDIT_SUM_DEBT_MEAN"], ((((data["CLOSED_AMT_CREDIT_SUM_DEBT_MEAN"]) - (data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]))) * 2.0)))))
    v["i453"] = 0.094650 * np.tanh((((((-1.0 * (((((data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]) > (
        (((((((((((data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]) * 2.0)) > (np.tanh((np.tanh((0.636620)))))) * 1.)) * 2.0)) * 2.0)) * 2.0))) * 1.))))) * 2.0)) * 2.0))
    v["i454"] = 0.097487 * np.tanh(((np.where(data["CLOSED_AMT_CREDIT_SUM_DEBT_MEAN"] < -99998, 0.318310, (-1.0 * (
        ((((((((3.141593) < (((data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]) + (((data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]) + (data["CLOSED_AMT_CREDIT_SUM_MEAN"])))))) * 1.)) * 2.0)) * 2.0)))))) / 2.0))
    v["i455"] = 0.099499 * np.tanh((((data["ACTIVE_DAYS_CREDIT_MEAN"]) > ((((((np.tanh((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]) / 2.0)))) > (np.where(
        data["ACTIVE_AMT_CREDIT_SUM_SUM"] > 0, 0.0, data["ACTIVE_AMT_CREDIT_SUM_SUM"]))) * 1.)) - (((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) - (data["ACTIVE_AMT_CREDIT_SUM_MEAN"])))))) * 1.))
    v["i456"] = 0.051000 * np.tanh(np.minimum(((np.maximum(((data["BURO_CREDIT_ACTIVE_Closed_MEAN"])), ((0.0))))), ((((np.minimum(
        ((((((((((-2.0) + (0.318310)) / 2.0)) / 2.0)) > (data["NEW_SCORES_STD"])) * 1.))), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) * (data["BURO_CREDIT_ACTIVE_Closed_MEAN"]))))))
    v["i457"] = 0.099734 * np.tanh((-1.0 * (((((np.where(np.maximum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((data["ACTIVE_DAYS_CREDIT_VAR"]))) > 0, data[
                                   "NAME_INCOME_TYPE_Student"], (((-1.0 * ((data["NEW_SCORES_STD"])))) / 2.0))) < (((-2.0) + (data["NEW_SCORES_STD"])))) * 1.)))))
    v["i458"] = 0.069980 * np.tanh((-1.0 * (((((data["AMT_CREDIT"]) > (np.where(((data["NAME_INCOME_TYPE_Student"]) + (data["AMT_CREDIT"])) > 0, np.where(
        data["YEARS_BUILD_AVG"] > 0, 2.0, ((data["AMT_CREDIT"]) + (data["AMT_CREDIT"]))), data["NAME_INCOME_TYPE_Student"]))) * 1.)))))
    v["i459"] = 0.042509 * np.tanh(((((((((data["AMT_ANNUITY"]) + (np.minimum(((data["ACTIVE_DAYS_CREDIT_VAR"])), (((((-1.0 * ((data["AMT_CREDIT"])))) - (
        data["AMT_CREDIT"]))))))) / 2.0)) > (np.maximum(((data["AMT_CREDIT"])), (((-1.0 * ((data["AMT_CREDIT"])))))))) * 1.)) * 2.0))
    v["i460"] = 0.096000 * np.tanh(((((((((((np.tanh((1.570796))) + (np.where(data["AMT_CREDIT"] > 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"],
                                                                              3.141593))) / 2.0)) < ((((np.tanh((1.0))) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.))) * 1.)) * 2.0)) * 2.0))
    v["i461"] = 0.004245 * np.tanh((-1.0 * ((((np.minimum(((data["AMT_CREDIT"])), ((data["AMT_ANNUITY"])))) * ((((data["YEARS_BUILD_AVG"]) > (
        ((((((-1.0) + ((((data["AMT_CREDIT"]) < (data["AMT_ANNUITY"])) * 1.))) / 2.0)) < (data["NEW_SCORES_STD"])) * 1.))) * 1.)))))))
    v["i462"] = 0.046750 * np.tanh(((data["AMT_CREDIT"]) * ((-1.0 * (((((data["AMT_CREDIT"]) > (np.where(
        data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, ((1.570796) + (2.0)), ((0.636620) * 2.0)))) * 1.)))))))
    v["i463"] = 0.099000 * np.tanh(np.minimum((((((np.maximum(((data["NEW_SCORES_STD"])), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) < (np.maximum(((data["CC_AMT_PAYMENT_CURRENT_VAR"])),
                                                                                                                                                  ((1.570796))))) * 1.))), ((((data["NEW_SCORES_STD"]) * ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < (data["CC_AMT_PAYMENT_CURRENT_VAR"])) * 1.)))))))
    v["i464"] = 0.090996 * np.tanh(((((((1.79046797752380371)) < (np.where((((((((2.0) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)) - (
        data["NAME_INCOME_TYPE_Student"]))) - (data["AMT_ANNUITY"])) < 0, (1.79046797752380371), data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) * 1.)) * (3.141593)))
    v["i465"] = 0.082002 * np.tanh(((np.where(data["ORGANIZATION_TYPE_Trade__type_3"] > 0, ((-2.0) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"])), ((-1.0) + (
        (((((-2.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) < ((-1.0 * ((data["ORGANIZATION_TYPE_Trade__type_3"]))))) * 1.))))) * 2.0))
    v["i466"] = 0.099646 * np.tanh((((-1.0 * ((((data["AMT_ANNUITY"]) * (np.maximum(((data["YEARS_BUILD_AVG"])), ((np.minimum(
        ((((np.tanh(((((1.570796) + (data["YEARS_BUILD_AVG"])) / 2.0)))) * (data["ORGANIZATION_TYPE_Trade__type_3"])))), ((data["ORGANIZATION_TYPE_Trade__type_3"])))))))))))) / 2.0))
    v["i467"] = 0.099820 * np.tanh((((((((((3.0) - ((-1.0 * (((((data["AMT_ANNUITY"]) < (((0.318310) / 2.0))) * 1.))))))) - (
        data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) < ((((1.570796) > (data["AMT_ANNUITY"])) * 1.))) * 1.)) * 2.0))
    v["i468"] = 0.097001 * np.tanh(((np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] < -99998, data["YEARS_BUILD_MODE"], 3.0)) * (
        (((((2.0) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)) - ((((((((3.86910891532897949)) + (0.318310)) / 2.0)) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.))))))
    v["i469"] = 0.099440 * np.tanh(np.where(np.where(data["NONLIVINGAREA_MEDI"] < 0, (((((data["NONLIVINGAREA_MEDI"]) < ((((data["NONLIVINGAREA_MODE"]) + (
        (-1.0 * ((data["YEARS_BUILD_MEDI"]))))) / 2.0))) * 1.)) - (data["NONLIVINGAREA_MODE"])), data["NONLIVINGAREA_MODE"]) > 0, 0.0, (10.0)))
    v["i470"] = 0.099890 * np.tanh((((((data["YEARS_BUILD_MODE"]) > ((((((data["YEARS_BUILD_MODE"]) > (data["NEW_ANNUITY_TO_INCOME_RATIO"])) * 1.)) * 2.0))) * 1.)) * (
        (((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (data["NONLIVINGAREA_MODE"])) * 1.)) - ((((data["NONLIVINGAREA_MODE"]) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.))))))
    v["i471"] = 0.002000 * np.tanh(np.where(data["YEARS_BUILD_AVG"] < 0, np.where(data["COMMONAREA_MEDI"] < 0, 0.0, (-1.0 * (
        (3.0)))), np.tanh((np.where(data["LIVINGAPARTMENTS_MEDI"] > 0, 0.0, data["NONLIVINGAPARTMENTS_AVG"])))))
    v["i472"] = 0.094998 * np.tanh((-1.0 * (((((1.0) > (((np.where(data["LIVINGAPARTMENTS_AVG"] < 0, 1.0, ((data["YEARS_BUILD_MODE"]) + (
        np.minimum(((((data["YEARS_BUILD_MEDI"]) + (data["LIVINGAPARTMENTS_MEDI"])))), ((1.0))))))) * 2.0))) * 1.)))))
    v["i473"] = 0.052500 * np.tanh(np.where(data["ORGANIZATION_TYPE_Trade__type_3"] < 0, (-1.0 * (((((data["AMT_CREDIT"]) > (
        (((2.89727640151977539)) + (np.where(data["YEARS_BUILD_MEDI"] < 0, np.tanh((data["YEARS_BUILD_MEDI"])), -1.0))))) * 1.)))), data["YEARS_BUILD_AVG"]))
    v["i474"] = 0.095291 * np.tanh((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < (data["YEARS_BUILD_AVG"])) * 1.)) * (((((-1.0 * ((np.where(data["NONLIVINGAREA_MODE"] < -99998, data[
                                   "NEW_ANNUITY_TO_INCOME_RATIO"], data["YEARS_BUILD_MEDI"]))))) > (np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] > 0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], 0.318310))) * 1.))))
    v["i475"] = 0.084300 * np.tanh(((data["YEARS_BUILD_AVG"]) * ((((data["COMMONAREA_AVG"]) > (np.maximum(((np.maximum((((((data["YEARS_BUILD_MEDI"]) + (0.318310)) / 2.0))), ((
        data["LIVINGAPARTMENTS_MODE"]))))), ((np.where(data["YEARS_BUILD_MEDI"] > 0, data["LIVINGAPARTMENTS_MEDI"], data["LIVINGAPARTMENTS_MEDI"])))))) * 1.))))
    v["i476"] = 0.099654 * np.tanh((((((np.where((((data["AMT_ANNUITY"]) < (((((6.01972627639770508)) > (data["YEARS_BUILD_MODE"])) * 1.))) * 1.) > 0, data[
                                   "YEARS_BUILD_MODE"], data["NEW_CREDIT_TO_ANNUITY_RATIO"])) > (1.570796)) * 1.)) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))
    v["i477"] = 0.040049 * np.tanh((-1.0 * ((((data["AMT_CREDIT"]) * (((((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) / 2.0)) < ((((data["YEARS_BUILD_MEDI"]) < (
        np.where(data["LIVINGAPARTMENTS_AVG"] < 0, data["YEARS_BUILD_MODE"], data["YEARS_BUILD_AVG"]))) * 1.))) * 1.)) < (data["YEARS_BUILD_AVG"])) * 1.)))))))
    v["i478"] = 0.099720 * np.tanh(((data["AMT_ANNUITY"]) * ((((data["LIVINGAPARTMENTS_MODE"]) > ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (
        (((((data["LIVINGAPARTMENTS_MODE"]) + ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (data["YEARS_BUILD_AVG"])) * 1.)))) > (data["YEARS_BUILD_AVG"])) * 1.))) / 2.0))) * 1.))))
    v["i479"] = 0.091803 * np.tanh((-1.0 * ((np.where(data["AMT_CREDIT"] > 0, ((((((data["YEARS_BUILD_MEDI"]) > (-2.0)) * 1.)) > (
        (((data["AMT_CREDIT"]) < (data["YEARS_BUILD_MEDI"])) * 1.))) * 1.), (((0.636620) < (((data["AMT_ANNUITY"]) / 2.0))) * 1.))))))
    v["i480"] = 0.099545 * np.tanh((((((3.141593) * (np.maximum((((((((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < -99998, data["ELEVATORS_MODE"], data[
                                   "YEARS_BEGINEXPLUATATION_AVG"])) > (0.318310)) * 1.)) * 2.0))), (((-1.0 * ((data["ELEVATORS_MODE"]))))))))) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.))
    v["i481"] = 0.100000 * np.tanh((((((((data["INSTAL_DBD_MEAN"]) > (1.0)) * 1.)) - (((((((data["NAME_INCOME_TYPE_Student"]) + (((np.maximum(
        ((data["NAME_INCOME_TYPE_Student"])), ((((data["INSTAL_PAYMENT_PERC_MAX"]) * 2.0))))) * 2.0))) / 2.0)) < (data["INSTAL_PAYMENT_PERC_MEAN"])) * 1.)))) * 2.0))
    v["i482"] = 0.011980 * np.tanh(np.maximum((((((3.0) < (data["CC_CNT_DRAWINGS_CURRENT_MEAN"])) * 1.))), (((((((((((((data["CC_AMT_DRAWINGS_OTHER_CURRENT_SUM"]) < (
        data["CC_CNT_DRAWINGS_CURRENT_MEAN"])) * 1.)) > (np.tanh((data["CC_CNT_DRAWINGS_CURRENT_MAX"])))) * 1.)) < (data["CC_AMT_DRAWINGS_OTHER_CURRENT_SUM"])) * 1.)) * (3.0))))))
    v["i483"] = 0.099951 * np.tanh((((((((3.141593) < (((data["CC_AMT_BALANCE_MEAN"]) + (((data["CC_NAME_CONTRACT_STATUS_Signed_MEAN"]) + (data["CC_CNT_DRAWINGS_CURRENT_MEAN"]))))))
                                        * 1.)) * 2.0)) * (np.where(data["CC_AMT_TOTAL_RECEIVABLE_MEAN"] < 0, 1.570796, data["CC_CNT_DRAWINGS_CURRENT_MEAN"]))))
    v["i484"] = 0.056600 * np.tanh(np.where(((data["CLOSED_AMT_ANNUITY_MAX"]) * (np.minimum(((data["NAME_TYPE_SUITE_Family"])), ((data["WALLSMATERIAL_MODE_Stone__brick"])))))
                                            < 0, 0.318310, (-1.0 * (((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + ((-1.0 * ((0.318310)))))) > (1.570796)) * 1.))))))
    v["i485"] = 0.096150 * np.tanh(np.minimum(((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["NAME_EDUCATION_TYPE_Academic_degree"]))) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) * (
        data["ORGANIZATION_TYPE_Military"])))), (((((((np.tanh((data["ORGANIZATION_TYPE_Military"]))) - (1.0))) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)))))
    v["i486"] = 0.038246 * np.tanh(((np.where(np.where(data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"] < -99998, 2.0, np.minimum(((data["AMT_CREDIT"])), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))
                                              > 0, ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["AMT_CREDIT"])), data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])) * ((-1.0 * ((data["NAME_INCOME_TYPE_Student"]))))))
    v["i487"] = 0.094998 * np.tanh(np.where(data["ACTIVE_DAYS_CREDIT_VAR"] > 0, 0.0, (((((np.tanh(((3.0)))) * (
        ((np.tanh((3.0))) * ((((3.0)) / 2.0)))))) < ((-1.0 * ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))) * 1.)))
    v["i488"] = 0.079999 * np.tanh(np.minimum((((-1.0 * ((((((data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]) * ((((data["AMT_ANNUITY"]) > ((((1.22967505455017090)) * 2.0))) * 1.))))
                                                           * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))))), ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + ((1.22967505455017090)))))))
    v["i489"] = 0.079740 * np.tanh((-1.0 * (((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)))) < (
        ((data["AMT_CREDIT"]) + ((-1.0 * ((((3.0) + (np.minimum(((data["AMT_CREDIT"])), (((8.0))))))))))))) * 1.)))))
    v["i490"] = 0.047699 * np.tanh((-1.0 * ((((((((((data["NEW_CREDIT_TO_INCOME_RATIO"]) + (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) / 2.0)))) > (data["ACTIVE_DAYS_CREDIT_VAR"])) * 1.)) < (
        (((data["NAME_INCOME_TYPE_Student"]) < ((((data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]) + (data["NAME_INCOME_TYPE_Maternity_leave"])) / 2.0))) * 1.))) * 1.)))))
    v["i491"] = 0.058100 * np.tanh((((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < 0, data["AMT_ANNUITY"], data["NEW_CREDIT_TO_ANNUITY_RATIO"])) > (((np.where(
        data["NEW_ANNUITY_TO_INCOME_RATIO"] < 0, (((data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]) < (2.0)) * 1.), 2.0)) - (np.tanh((data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"])))))) * 1.))
    v["i492"] = 0.099590 * np.tanh(((np.minimum(((((0.636620) - (data["AMT_ANNUITY"])))), (((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (
        data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"])) / 2.0))) / 2.0))))) * ((((data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]) > (data["ORGANIZATION_TYPE_Trade__type_4"])) * 1.))))
    v["i493"] = 0.049969 * np.tanh((-1.0 * ((((data["AMT_ANNUITY"]) * (np.maximum(((data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"])), ((np.maximum(((data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"])),
                                                                                                                                               ((((np.where(data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"] < 0, data["ORGANIZATION_TYPE_Trade__type_4"], (-1.0 * ((data["AMT_ANNUITY"]))))) * 2.0)))))))))))))
    v["i494"] = 0.073170 * np.tanh((((((((2.0) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)) * 2.0)) - (
        (((((((((((3.141593) - (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)))) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) * 1.)) * 2.0)) * 2.0)) * 2.0))))
    v["i495"] = 0.099300 * np.tanh((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (((2.0) + (
        (((np.tanh((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)))) > (((1.570796) / 2.0))) * 1.))))) * 1.)) - (data["ORGANIZATION_TYPE_Trade__type_4"]))) * 2.0))
    v["i496"] = 0.096644 * np.tanh((-1.0 * ((((((data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"]) * ((((np.where(data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"] > 0, ((
        data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"]) - (2.0)), data["NEW_CREDIT_TO_ANNUITY_RATIO"])) > (2.0)) * 1.)))) * (((data["AMT_ANNUITY"]) / 2.0)))))))
    v["i497"] = 0.099799 * np.tanh(((data["ORGANIZATION_TYPE_Restaurant"]) * (((data["ORGANIZATION_TYPE_Restaurant"]) * (
        ((((np.maximum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((((0.318310) - (np.maximum(((3.0)), ((1.570796))))))))) / 2.0)) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))))
    v["i498"] = 0.099000 * np.tanh(np.where(data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"] > 0, ((((((data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"]) / 2.0)) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) * (
        data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"])), ((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (((3.0) / 2.0))) / 2.0)) < (data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"])) * 1.)))
    v["i499"] = 0.094680 * np.tanh(np.where(data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"] < 0, (-1.0 * (((((2.0) < ((((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (np.tanh(
        (data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"])))) / 2.0)) + (data["ORGANIZATION_TYPE_Industry__type_3"]))) * 2.0))) * 1.)))), data["WEEKDAY_APPR_PROCESS_START_WEDNESDAY"]))
    v["i500"] = 0.002300 * np.tanh((-1.0 * ((np.where((((((data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]) < (((data["AMT_ANNUITY"]) * (data["NEW_ANNUITY_TO_INCOME_RATIO"])))) * 1.)) - (np.where(
        data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"] < 0, data["AMT_ANNUITY"], data["NEW_ANNUITY_TO_INCOME_RATIO"]))) > 0, data["NAME_INCOME_TYPE_Student"], data["WEEKDAY_APPR_PROCESS_START_WEDNESDAY"])))))
    v["i501"] = 0.017500 * np.tanh((((((((((data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]) + (data["WEEKDAY_APPR_PROCESS_START_WEDNESDAY"])) / 2.0)) > (data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"])) * 1.)) < (
        ((((((data["WEEKDAY_APPR_PROCESS_START_WEDNESDAY"]) > (data["AMT_ANNUITY"])) * 1.)) < ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"])) * 1.))) * 1.))) * 1.))
    v["i502"] = 0.099500 * np.tanh(((((np.minimum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), ((data["NEW_ANNUITY_TO_INCOME_RATIO"])))) - (data["AMT_ANNUITY"]))) * (np.where(
        ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["AMT_ANNUITY"])) > 0, (((data["NAME_INCOME_TYPE_Maternity_leave"]) < (data["WEEKDAY_APPR_PROCESS_START_WEDNESDAY"])) * 1.), data["REG_REGION_NOT_LIVE_REGION"]))))
    v["i503"] = 0.012550 * np.tanh((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > (((np.where(data["AMT_ANNUITY"] < 0, np.where(
        data["WALLSMATERIAL_MODE_Wooden"] < 0, ((1.570796) + (2.0)), data["NEW_CREDIT_TO_ANNUITY_RATIO"]), (5.0))) / 2.0))) * 1.))
    v["i504"] = 0.097100 * np.tanh(((-2.0) * ((((3.141593) < (np.maximum(((data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"])), ((((np.where(data["NAME_FAMILY_STATUS_Single___not_married"]
                                                                                                                                  < 0, np.maximum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), ((data["ORGANIZATION_TYPE_Industry__type_13"]))), data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"])) - (-1.0))))))) * 1.))))
    v["i505"] = 0.090200 * np.tanh(((((((((((((data["AMT_CREDIT"]) * (data["AMT_ANNUITY"]))) > (((3.0) * (
        data["AMT_ANNUITY"])))) * 1.)) / 2.0)) / 2.0)) > (((2.0) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) * 1.))
    v["i506"] = 0.076249 * np.tanh(np.where((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (-2.0)) / 2.0) > 0, np.where(data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"] > 0, data["NEW_ANNUITY_TO_INCOME_RATIO"], data[
                                   "BURO_AMT_CREDIT_SUM_LIMIT_MEAN"]), (((np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] > 0, data["AMT_ANNUITY"], data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"])) > (1.570796)) * 1.)))
    v["i507"] = 0.099600 * np.tanh(((((((data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]) > (data["ORGANIZATION_TYPE_Trade__type_4"])) * 1.)) > ((((np.where(np.where(data["ORGANIZATION_TYPE_Trade__type_4"]
                                                                                                                                                                 < 0, np.tanh((data["AMT_ANNUITY"])), data["NEW_ANNUITY_TO_INCOME_RATIO"]) < 0, data["NEW_ANNUITY_TO_INCOME_RATIO"], data["AMT_ANNUITY"])) < (2.0)) * 1.))) * 1.))
    v["i508"] = 0.008489 * np.tanh(((((np.minimum(((np.where(data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"] < -99998, data["NAME_INCOME_TYPE_Student"], (0.01891136541962624)))), ((
        ((np.where(data["WALLSMATERIAL_MODE_Stone__brick"] < 0, (-1.0 * ((data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]))), 1.570796)) - (data["NAME_INCOME_TYPE_Student"])))))) * 2.0)) * 2.0))
    v["i509"] = 0.092789 * np.tanh(((((np.where(data["AMT_ANNUITY"] < -99998, data["NEW_ANNUITY_TO_INCOME_RATIO"], ((data["CODE_GENDER"]) * (np.maximum(
        (((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < ((-1.0 * ((1.570796))))) * 1.))), ((data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]))))))) * 2.0)) * 2.0))
    v["i510"] = 0.069406 * np.tanh((((((((np.where(((np.where(data["NEW_EXT_SOURCES_MEAN"] > 0, data["NEW_ANNUITY_TO_INCOME_RATIO"], 0.636620)) + (
        data["NEW_ANNUITY_TO_INCOME_RATIO"])) > 0, data["NEW_ANNUITY_TO_INCOME_RATIO"], data["NEW_CREDIT_TO_ANNUITY_RATIO"])) - (data["NEW_EXT_SOURCES_MEAN"]))) > (3.0)) * 1.)) * 2.0))
    v["i511"] = 0.093680 * np.tanh((((((np.where(data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"] < -99998, data["EXT_SOURCE_3"], np.where(data["EXT_SOURCE_3"] < -99998, data["AMT_ANNUITY"], ((
        (data["NAME_INCOME_TYPE_Maternity_leave"]) < (data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"])) * 1.)))) > ((((data["AMT_ANNUITY"]) > (data["EXT_SOURCE_3"])) * 1.))) * 1.)) * 2.0))
    return Output(v.sum(axis=1))


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-hot encoding for categorical columns with get_dummies


def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [
        col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns,
                        dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Preprocess application_train.csv and application_test.csv


def application_train_test(num_rows=None, nan_as_category=False):
    # Read data and merge
    df = pd.read_csv('../input/application_train.csv', nrows=num_rows)
    test_df = pd.read_csv('../input/application_test.csv', nrows=num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']

    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in df.columns if ('FLAG_' in _f) & (
        'FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby(
        'ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']

    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
    df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
    df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_ANNUITY_TO_INCOME_RATIO'] = df[
        'AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
    df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * \
        df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['NEW_EXT_SOURCES_MEAN'] = df[
        ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_SCORES_STD'] = df[
        ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(
        df['NEW_SCORES_STD'].mean())
    df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df[
        'DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER'] = df[
        'DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['NEW_CREDIT_TO_INCOME_RATIO'] = df[
        'AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    dropcolum = ['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4',
                 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7',
                 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10',
                 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13',
                 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
                 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19',
                 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']
    df = df.drop(dropcolum, axis=1)
    del test_df
    gc.collect()
    return df

# Preprocess bureau.csv and bureau_balance.csv


def bureau_and_balance(num_rows=None, nan_as_category=True):
    bureau = pd.read_csv('../input/bureau.csv', nrows=num_rows)
    bb = pd.read_csv('../input/bureau_balance.csv', nrows=num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper()
                               for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat:
        cat_aggregations[cat] = ['mean']
    for cat in bb_cat:
        cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(
        ['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(
        ['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(
        ['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

# Preprocess previous_applications.csv


def previous_applications(num_rows=None, nan_as_category=True):
    prev = pd.read_csv('../input/previous_application.csv', nrows=num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['max', 'mean'],
        'AMT_APPLICATION': ['max', 'mean'],
        'AMT_CREDIT': ['max', 'mean'],
        'APP_CREDIT_PERC': ['max', 'mean'],
        'AMT_DOWN_PAYMENT': ['max', 'mean'],
        'AMT_GOODS_PRICE': ['max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['max', 'mean'],
        'RATE_DOWN_PAYMENT': ['max', 'mean'],
        'DAYS_DECISION': ['max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(
        ['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(
        ['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(
        ['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

# Preprocess POS_CASH_balance.csv


def pos_cash(num_rows=None, nan_as_category=True):
    pos = pd.read_csv('../input/POS_CASH_balance.csv', nrows=num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(
        ['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg

# Preprocess installments_payments.csv


def installments_payments(num_rows=None, nan_as_category=True):
    ins = pd.read_csv('../input/installments_payments.csv', nrows=num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)
    # Percentage and difference paid in each installment (amount paid and
    # installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum', 'min', 'std'],
        'DBD': ['max', 'mean', 'sum', 'min', 'std'],
        'PAYMENT_PERC': ['max', 'mean',  'var', 'min', 'std'],
        'PAYMENT_DIFF': ['max', 'mean', 'var', 'min', 'std'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum', 'min', 'std'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum', 'std'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum', 'std']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(
        ['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv


def credit_card_balance(num_rows=None, nan_as_category=True):
    cc = pd.read_csv('../input/credit_card_balance.csv', nrows=num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper()
                               for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg

debug = None
num_rows = 10000 if debug else None
df = application_train_test(num_rows)
with timer("Process bureau and bureau_balance"):
    bureau = bureau_and_balance(num_rows)
    print("Bureau df shape:", bureau.shape)
    df = df.join(bureau, how='left', on='SK_ID_CURR')
    del bureau
    gc.collect()
with timer("Process previous_applications"):
    prev = previous_applications(num_rows)
    print("Previous applications df shape:", prev.shape)
    df = df.join(prev, how='left', on='SK_ID_CURR')
    del prev
    gc.collect()
with timer("Process POS-CASH balance"):
    pos = pos_cash(num_rows)
    print("Pos-cash balance df shape:", pos.shape)
    df = df.join(pos, how='left', on='SK_ID_CURR')
    del pos
    gc.collect()
with timer("Process installments payments"):
    ins = installments_payments(num_rows)
    print("Installments payments df shape:", ins.shape)
    df = df.join(ins, how='left', on='SK_ID_CURR')
    del ins
    gc.collect()
with timer("Process credit card balance"):
    cc = credit_card_balance(num_rows)
    print("Credit card balance df shape:", cc.shape)
    df = df.join(cc, how='left', on='SK_ID_CURR')
    del cc
    gc.collect()

feats = [f for f in df.columns if f not in [
    'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

for c in feats:
    print(c)
    ss = StandardScaler()
    df.loc[~np.isfinite(df[c]), c] = np.nan
    df.loc[~df[c].isnull(), c] = ss.fit_transform(
        df.loc[~df[c].isnull(), c].values.reshape(-1, 1))
    df[c].fillna(-99999., inplace=True)

train_df = df[df['TARGET'].notnull()]
test_df = df[df['TARGET'].isnull()]
print(train_df.shape)
print(test_df.shape)

train_df.columns = train_df.columns.str.replace('[^A-Za-z0-9_]', '_')
test_df.columns = test_df.columns.str.replace('[^A-Za-z0-9_]', '_')
feats = [f for f in train_df.columns if f not in [
    'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

sub = test_df[['SK_ID_CURR']].copy()
sub['TARGET'] = (GP1(test_df.round(6)) +
                 GP2(test_df.round(6)) +
                 GP3(test_df.round(6))) / 3
sub.to_csv('gpalldatasubmission.csv', index=False)
