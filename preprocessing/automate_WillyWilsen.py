import joblib
import numpy as np
import pandas as pd

encoder_Application_mode = joblib.load("preprocessing/model/encoder_Application_mode.joblib")
encoder_Course = joblib.load("preprocessing/model/encoder_Course.joblib")
encoder_Daytime_evening_attendance = joblib.load("preprocessing/model/encoder_Daytime_evening_attendance.joblib")
encoder_Debtor = joblib.load("preprocessing/model/encoder_Debtor.joblib")
encoder_Displaced = joblib.load("preprocessing/model/encoder_Displaced.joblib")
encoder_Educational_special_needs = joblib.load("preprocessing/model/encoder_Educational_special_needs.joblib")
encoder_Fathers_occupation = joblib.load("preprocessing/model/encoder_Fathers_occupation.joblib")
encoder_Fathers_qualification = joblib.load("preprocessing/model/encoder_Fathers_qualification.joblib")
encoder_Gender = joblib.load("preprocessing/model/encoder_Gender.joblib")
encoder_International = joblib.load("preprocessing/model/encoder_International.joblib")
encoder_Marital_status = joblib.load("preprocessing/model/encoder_Marital_status.joblib")
encoder_Mothers_occupation = joblib.load("preprocessing/model/encoder_Mothers_occupation.joblib")
encoder_Mothers_qualification = joblib.load("preprocessing/model/encoder_Mothers_qualification.joblib")
encoder_Nacionality = joblib.load("preprocessing/model/encoder_Nacionality.joblib")
encoder_Previous_qualification = joblib.load("preprocessing/model/encoder_Previous_qualification.joblib")
encoder_Scholarship_holder = joblib.load("preprocessing/model/encoder_Scholarship_holder.joblib")
encoder_Tuition_fees_up_to_date = joblib.load("preprocessing/model/encoder_Tuition_fees_up_to_date.joblib")
pca_1 = joblib.load("preprocessing/model/pca_1.joblib")
scaler_Admission_grade = joblib.load("preprocessing/model/scaler_Admission_grade.joblib")
scaler_Age_at_enrollment = joblib.load("preprocessing/model/scaler_Age_at_enrollment.joblib")
scaler_Application_order = joblib.load("preprocessing/model/scaler_Application_order.joblib")
scaler_Curricular_units_1st_sem_approved = joblib.load("preprocessing/model/scaler_Curricular_units_1st_sem_approved.joblib")
scaler_Curricular_units_1st_sem_credited = joblib.load("preprocessing/model/scaler_Curricular_units_1st_sem_credited.joblib")
scaler_Curricular_units_1st_sem_enrolled = joblib.load("preprocessing/model/scaler_Curricular_units_1st_sem_enrolled.joblib")
scaler_Curricular_units_1st_sem_evaluations = joblib.load("preprocessing/model/scaler_Curricular_units_1st_sem_evaluations.joblib")
scaler_Curricular_units_1st_sem_grade = joblib.load("preprocessing/model/scaler_Curricular_units_1st_sem_grade.joblib")
scaler_Curricular_units_1st_sem_without_evaluations = joblib.load("preprocessing/model/scaler_Curricular_units_1st_sem_without_evaluations.joblib")
scaler_Curricular_units_2nd_sem_approved = joblib.load("preprocessing/model/scaler_Curricular_units_2nd_sem_approved.joblib")
scaler_Curricular_units_2nd_sem_credited = joblib.load("preprocessing/model/scaler_Curricular_units_2nd_sem_credited.joblib")
scaler_Curricular_units_2nd_sem_enrolled = joblib.load("preprocessing/model/scaler_Curricular_units_2nd_sem_enrolled.joblib")
scaler_Curricular_units_2nd_sem_evaluations = joblib.load("preprocessing/model/scaler_Curricular_units_2nd_sem_evaluations.joblib")
scaler_Curricular_units_2nd_sem_grade = joblib.load("preprocessing/model/scaler_Curricular_units_2nd_sem_grade.joblib")
scaler_Curricular_units_2nd_sem_without_evaluations = joblib.load("preprocessing/model/scaler_Curricular_units_2nd_sem_without_evaluations.joblib")
scaler_GDP = joblib.load("preprocessing/model/scaler_GDP.joblib")
scaler_Inflation_rate = joblib.load("preprocessing/model/scaler_Inflation_rate.joblib")
scaler_Previous_qualification_grade = joblib.load("preprocessing/model/scaler_Previous_qualification_grade.joblib")
scaler_Unemployment_rate = joblib.load("preprocessing/model/scaler_Unemployment_rate.joblib")

encoder_target = joblib.load("preprocessing/model/encoder_target.joblib")

pca_numerical_columns_1 = [
    'Curricular_units_1st_sem_credited',
    'Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_evaluations',
    'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade',
    'Curricular_units_2nd_sem_credited',
    'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_evaluations',
    'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade'
]

def data_preprocessing(data):
    """Preprocessing data
 
    Args:
        data (Pandas DataFrame): Dataframe that contain all the data to make prediction 
        
    return:
        Pandas DataFrame: Dataframe that contain all the preprocessed data
    """
    data = data.copy()

    marital_status_map = {
        1: 'single',
        2: 'married',
        3: 'widower',
        4: 'divorced',
        5: 'facto union',
        6: 'legally separated'
    }

    if pd.api.types.is_numeric_dtype(data['Marital_status']):
        data['Marital_status'] = data['Marital_status'].map(marital_status_map)

    application_mode_map = {
        1: '1st phase - general contingent',
        2: 'Ordinance No. 612/93',
        5: '1st phase - special contingent (Azores Island)',
        7: 'Holders of other higher courses',
        10: 'Ordinance No. 854-B/99',
        15: 'International student (bachelor)',
        16: '1st phase - special contingent (Madeira Island)',
        17: '2nd phase - general contingent',
        18: '3rd phase - general contingent',
        26: 'Ordinance No. 533-A/99, item b2) (Different Plan)',
        27: 'Ordinance No. 533-A/99, item b3 (Other Institution)',
        39: 'Over 23 years old',
        42: 'Transfer',
        43: 'Change of course',
        44: 'Technological specialization diploma holders',
        51: 'Change of institution/course',
        53: 'Short cycle diploma holders',
        57: 'Change of institution/course (International)'
    }

    if pd.api.types.is_numeric_dtype(data['Application_mode']):
        data['Application_mode'] = data['Application_mode'].map(application_mode_map)

    course_map = {
        33: "Biofuel Production Technologies",
        171: "Animation and Multimedia Design",
        8014: "Social Service (evening attendance)",
        9003: "Agronomy",
        9070: "Communication Design",
        9085: "Veterinary Nursing",
        9119: "Informatics Engineering",
        9130: "Equinculture",
        9147: "Management",
        9238: "Social Service",
        9254: "Tourism",
        9500: "Nursing",
        9556: "Oral Hygiene",
        9670: "Advertising and Marketing Management",
        9773: "Journalism and Communication",
        9853: "Basic Education",
        9991: "Management (evening attendance)"
    }

    if pd.api.types.is_numeric_dtype(data['Course']):
        data['Course'] = data['Course'].map(course_map)

    attendance_map = {
        1: 'daytime',
        0: 'evening'
    }

    if pd.api.types.is_numeric_dtype(data['Daytime_evening_attendance']):
        data['Daytime_evening_attendance'] = data['Daytime_evening_attendance'].map(attendance_map)

    previous_qualification_map = {
        1: "Secondary education",
        2: "Higher education - bachelor's degree",
        3: "Higher education - degree",
        4: "Higher education - master's",
        5: "Higher education - doctorate",
        6: "Frequency of higher education",
        9: "12th year of schooling - not completed",
        10: "11th year of schooling - not completed",
        12: "Other - 11th year of schooling",
        14: "10th year of schooling",
        15: "10th year of schooling - not completed",
        19: "Basic education 3rd cycle (9th/10th/11th year) or equiv.",
        38: "Basic education 2nd cycle (6th/7th/8th year) or equiv.",
        39: "Technological specialization course",
        40: "Higher education - degree (1st cycle)",
        42: "Professional higher technical course",
        43: "Higher education - master (2nd cycle)"
    }

    if pd.api.types.is_numeric_dtype(data['Previous_qualification']):
        data['Previous_qualification'] = data['Previous_qualification'].map(previous_qualification_map)

    nacionality_map = {
        1: "Portuguese",
        2: "German",
        6: "Spanish",
        11: "Italian",
        13: "Dutch",
        14: "English",
        17: "Lithuanian",
        21: "Angolan",
        22: "Cape Verdean",
        24: "Guinean",
        25: "Mozambican",
        26: "Santomean",
        32: "Turkish",
        41: "Brazilian",
        62: "Romanian",
        100: "Moldova (Republic of)",
        101: "Mexican",
        103: "Ukrainian",
        105: "Russian",
        108: "Cuban",
        109: "Colombian"
    }

    if pd.api.types.is_numeric_dtype(data['Nacionality']):
        data['Nacionality'] = data['Nacionality'].map(nacionality_map)

    mothers_qualification_map = {
        1: "Secondary Education - 12th Year or Eq.",
        2: "Higher Education - Bachelor's Degree",
        3: "Higher Education - Degree",
        4: "Higher Education - Master's",
        5: "Higher Education - Doctorate",
        6: "Frequency of Higher Education",
        9: "12th Year - Not Completed",
        10: "11th Year - Not Completed",
        11: "7th Year (Old)",
        12: "Other - 11th Year",
        14: "10th Year",
        18: "General Commerce Course",
        19: "Basic Education 3rd Cycle",
        22: "Technical-professional Course",
        26: "7th Year of Schooling",
        27: "2nd Cycle of General High School",
        29: "9th Year - Not Completed",
        30: "8th Year",
        34: "Unknown",
        35: "Can't Read or Write",
        36: "Can Read Without 4th Year",
        37: "Basic Education 1st Cycle",
        38: "Basic Education 2nd Cycle",
        39: "Technological Specialization Course",
        40: "Higher Education - Degree (1st Cycle)",
        41: "Specialized Higher Studies Course",
        42: "Professional Higher Technical Course",
        43: "Higher Education - Master (2nd Cycle)",
        44: "Higher Education - Doctorate (3rd Cycle)"
    }

    if pd.api.types.is_numeric_dtype(data['Mothers_qualification']):
        data['Mothers_qualification'] = data['Mothers_qualification'].map(mothers_qualification_map)

    fathers_qualification_map = {
        1: "Secondary Education - 12th Year or Eq.",
        2: "Higher Education - Bachelor's Degree",
        3: "Higher Education - Degree",
        4: "Higher Education - Master's",
        5: "Higher Education - Doctorate",
        6: "Frequency of Higher Education",
        9: "12th Year - Not Completed",
        10: "11th Year - Not Completed",
        11: "7th Year (Old)",
        12: "Other - 11th Year",
        13: "2nd Year Complementary High School",
        14: "10th Year",
        18: "General Commerce Course",
        19: "Basic Education 3rd Cycle",
        20: "Complementary High School Course",
        22: "Technical-professional Course",
        25: "Complementary High School Course - Not Concluded",
        26: "7th Year of Schooling",
        27: "2nd Cycle of General High School",
        29: "9th Year - Not Completed",
        30: "8th Year",
        31: "General Course of Administration and Commerce",
        33: "Supplementary Accounting and Administration",
        34: "Unknown",
        35: "Can't Read or Write",
        36: "Can Read Without 4th Year",
        37: "Basic Education 1st Cycle",
        38: "Basic Education 2nd Cycle",
        39: "Technological Specialization Course",
        40: "Higher Education - Degree (1st Cycle)",
        41: "Specialized Higher Studies Course",
        42: "Professional Higher Technical Course",
        43: "Higher Education - Master (2nd Cycle)",
        44: "Higher Education - Doctorate (3rd Cycle)"
    }

    if pd.api.types.is_numeric_dtype(data['Fathers_qualification']):
        data['Fathers_qualification'] = data['Fathers_qualification'].map(fathers_qualification_map)

    mothers_occupation_mapping = {
        0: "Student",
        1: "Legislative/Executive Representatives, Directors",
        2: "Intellectual and Scientific Specialists",
        3: "Intermediate Level Technicians",
        4: "Administrative Staff",
        5: "Personal Services, Security, Sellers",
        6: "Farmers, Agriculture, Fisheries",
        7: "Industry/Construction Craftsmen",
        8: "Machine Operators and Assemblers",
        9: "Unskilled Workers",
        10: "Armed Forces Professions",
        90: "Other Situation",
        99: "Blank",
        122: "Health Professionals",
        123: "Teachers",
        125: "ICT Specialists",
        131: "Intermediate Science & Engineering Technicians",
        132: "Intermediate Health Technicians",
        134: "Intermediate Legal, Social, Sports, Cultural Technicians",
        141: "Office Workers, Secretaries, Data Operators",
        143: "Accounting, Finance, Registry Operators",
        144: "Other Admin Support Staff",
        151: "Personal Service Workers",
        152: "Sellers",
        153: "Personal Care Workers",
        171: "Skilled Construction Workers (except Electricians)",
        173: "Skilled Workers in Printing, Instruments, Jewelers, Artisans",
        175: "Workers in Food, Woodworking, Clothing & Other Industries",
        191: "Cleaning Workers",
        192: "Unskilled Agriculture, Animal Production, Fisheries",
        193: "Unskilled Extractive, Construction, Manufacturing, Transport",
        194: "Meal Preparation Assistants"
    }

    if pd.api.types.is_numeric_dtype(data['Mothers_occupation']):
        data['Mothers_occupation'] = data['Mothers_occupation'].map(mothers_occupation_mapping)

    fathers_occupation_mapping = {
        0: "Student",
        1: "Legislative/Executive Representatives, Directors",
        2: "Intellectual and Scientific Specialists",
        3: "Intermediate Level Technicians",
        4: "Administrative Staff",
        5: "Personal Services, Security, Sellers",
        6: "Farmers and Skilled Workers in Agriculture, Fisheries, Forestry",
        7: "Skilled Industry, Construction, Craftsmen",
        8: "Machine Operators and Assemblers",
        9: "Unskilled Workers",
        10: "Armed Forces Professions",
        90: "Other Situation",
        99: "Blank",
        101: "Armed Forces Officers",
        102: "Armed Forces Sergeants",
        103: "Other Armed Forces Personnel",
        112: "Directors - Admin and Commercial Services",
        114: "Directors - Hotel, Catering, Trade, Other Services",
        121: "Specialists - Physical Sciences, Math, Engineering",
        122: "Health Professionals",
        123: "Teachers",
        124: "Finance, Accounting, Admin, Public/Commercial Relations Specialists",
        131: "Intermediate Science & Engineering Technicians",
        132: "Intermediate Health Technicians",
        134: "Intermediate Legal, Social, Sports, Cultural Technicians",
        135: "ICT Technicians",
        141: "Office Workers, Secretaries, Data Operators",
        143: "Accounting, Statistical, Financial, Registry Operators",
        144: "Other Admin Support Staff",
        151: "Personal Service Workers",
        152: "Sellers",
        153: "Personal Care Workers",
        154: "Protection and Security Personnel",
        161: "Market-Oriented Farmers, Skilled Agriculture/Animal Production",
        163: "Subsistence Farmers, Fishermen, Hunters",
        171: "Skilled Construction Workers (except Electricians)",
        172: "Skilled Metal, Metallurgy Workers",
        174: "Skilled Electrical/Electronic Workers",
        175: "Workers in Food, Wood, Clothing, Crafts",
        181: "Fixed Plant and Machine Operators",
        182: "Assembly Workers",
        183: "Vehicle Drivers and Mobile Equipment Operators",
        192: "Unskilled Workers in Agriculture/Fisheries",
        193: "Unskilled Workers in Extractive, Construction, Manufacturing, Transport",
        194: "Meal Preparation Assistants",
        195: "Street Vendors (Non-food) and Street Services"
    }

    if pd.api.types.is_numeric_dtype(data['Fathers_occupation']):
        data['Fathers_occupation'] = data['Fathers_occupation'].map(fathers_occupation_mapping)

    displaced_mapping = {
        1: "Yes",
        0: "No"
    }

    if pd.api.types.is_numeric_dtype(data['Displaced']):
        data['Displaced'] = data['Displaced'].map(displaced_mapping)

    special_needs_mapping = {
        1: "Yes",
        0: "No"
    }

    if pd.api.types.is_numeric_dtype(data['Educational_special_needs']):
        data['Educational_special_needs'] = data['Educational_special_needs'].map(special_needs_mapping)

    debtor_mapping = {
        1: "Yes",
        0: "No"
    }

    if pd.api.types.is_numeric_dtype(data['Debtor']):
        data['Debtor'] = data['Debtor'].map(debtor_mapping)

    tuition_fees_mapping = {
        1: "Yes",
        0: "No"
    }

    if pd.api.types.is_numeric_dtype(data['Tuition_fees_up_to_date']):
        data['Tuition_fees_up_to_date'] = data['Tuition_fees_up_to_date'].map(tuition_fees_mapping)

    gender_mapping = {
        1: "Male",
        0: "Female"
    }

    if pd.api.types.is_numeric_dtype(data['Gender']):
        data['Gender'] = data['Gender'].map(gender_mapping)

    scholarship_mapping = {
        1: "Yes",
        0: "No"
    }

    if pd.api.types.is_numeric_dtype(data['Scholarship_holder']):
        data['Scholarship_holder'] = data['Scholarship_holder'].map(scholarship_mapping)

    international_mapping = {
        1: "Yes",
        0: "No"
    }

    if pd.api.types.is_numeric_dtype(data['International']):
        data['International'] = data['International'].map(international_mapping)
    
    df = pd.DataFrame()

    df["Application_mode"] = encoder_Application_mode.transform(data["Application_mode"])
    df["Course"] = encoder_Course.transform(data["Course"])
    df["Daytime_evening_attendance"] = encoder_Daytime_evening_attendance.transform(data["Daytime_evening_attendance"])
    df["Debtor"] = encoder_Debtor.transform(data["Debtor"])
    df["Displaced"] = encoder_Displaced.transform(data["Displaced"])
    df["Educational_special_needs"] = encoder_Educational_special_needs.transform(data["Educational_special_needs"])
    df["Fathers_occupation"] = encoder_Fathers_occupation.transform(data["Fathers_occupation"])
    df["Fathers_qualification"] = encoder_Fathers_qualification.transform(data["Fathers_qualification"])
    df["Gender"] = encoder_Gender.transform(data["Gender"])
    df["International"] = encoder_International.transform(data["International"])
    df["Marital_status"] = encoder_Marital_status.transform(data["Marital_status"])
    df["Mothers_occupation"] = encoder_Mothers_occupation.transform(data["Mothers_occupation"])
    df["Mothers_qualification"] = encoder_Mothers_qualification.transform(data["Mothers_qualification"])
    df["Nacionality"] = encoder_Nacionality.transform(data["Nacionality"])
    df["Previous_qualification"] = encoder_Previous_qualification.transform(data["Previous_qualification"])
    df["Scholarship_holder"] = encoder_Scholarship_holder.transform(data["Scholarship_holder"])
    df["Tuition_fees_up_to_date"] = encoder_Tuition_fees_up_to_date.transform(data["Tuition_fees_up_to_date"])

    df["Admission_grade"] = scaler_Admission_grade.transform(np.asarray(data["Admission_grade"]).reshape(-1,1))
    df["Age_at_enrollment"] = scaler_Age_at_enrollment.transform(np.asarray(data["Age_at_enrollment"]).reshape(-1,1))
    df["Application_order"] = scaler_Application_order.transform(np.asarray(data["Application_order"]).reshape(-1,1))
    df["Curricular_units_1st_sem_without_evaluations"] = scaler_Curricular_units_1st_sem_without_evaluations.transform(np.asarray(data["Curricular_units_1st_sem_without_evaluations"]).reshape(-1,1))
    df["Curricular_units_2nd_sem_without_evaluations"] = scaler_Curricular_units_2nd_sem_without_evaluations.transform(np.asarray(data["Curricular_units_2nd_sem_without_evaluations"]).reshape(-1,1))
    df["GDP"] = scaler_GDP.transform(np.asarray(data["GDP"]).reshape(-1,1))
    df["Inflation_rate"] = scaler_Inflation_rate.transform(np.asarray(data["Inflation_rate"]).reshape(-1,1))
    df["Previous_qualification_grade"] = scaler_Previous_qualification_grade.transform(np.asarray(data["Previous_qualification_grade"]).reshape(-1,1))
    df["Unemployment_rate"] = scaler_Unemployment_rate.transform(np.asarray(data["Unemployment_rate"]).reshape(-1,1))

    # PCA 1
    df["Curricular_units_1st_sem_credited"] = scaler_Curricular_units_1st_sem_credited.transform(np.asarray(data["Curricular_units_1st_sem_credited"]).reshape(-1,1))
    df["Curricular_units_1st_sem_enrolled"] = scaler_Curricular_units_1st_sem_enrolled.transform(np.asarray(data["Curricular_units_1st_sem_enrolled"]).reshape(-1,1))
    df["Curricular_units_1st_sem_evaluations"] = scaler_Curricular_units_1st_sem_evaluations.transform(np.asarray(data["Curricular_units_1st_sem_evaluations"]).reshape(-1,1))
    df["Curricular_units_1st_sem_approved"] = scaler_Curricular_units_1st_sem_approved.transform(np.asarray(data["Curricular_units_1st_sem_approved"]).reshape(-1,1))
    df["Curricular_units_1st_sem_grade"] = scaler_Curricular_units_1st_sem_grade.transform(np.asarray(data["Curricular_units_1st_sem_grade"]).reshape(-1,1))
    df["Curricular_units_2nd_sem_credited"] = scaler_Curricular_units_2nd_sem_credited.transform(np.asarray(data["Curricular_units_2nd_sem_credited"]).reshape(-1,1))
    df["Curricular_units_2nd_sem_enrolled"] = scaler_Curricular_units_2nd_sem_enrolled.transform(np.asarray(data["Curricular_units_2nd_sem_enrolled"]).reshape(-1,1))
    df["Curricular_units_2nd_sem_evaluations"] = scaler_Curricular_units_2nd_sem_evaluations.transform(np.asarray(data["Curricular_units_2nd_sem_evaluations"]).reshape(-1,1))
    df["Curricular_units_2nd_sem_approved"] = scaler_Curricular_units_2nd_sem_approved.transform(np.asarray(data["Curricular_units_2nd_sem_approved"]).reshape(-1,1))
    df["Curricular_units_2nd_sem_grade"] = scaler_Curricular_units_2nd_sem_grade.transform(np.asarray(data["Curricular_units_2nd_sem_grade"]).reshape(-1,1))
    
    df[["pc1_1", "pc1_2", "pc1_3"]] = pca_1.transform(data[pca_numerical_columns_1])

    # Encode target variable
    df["Status"] = encoder_target.transform(data["Status"])

    desired_order = [
        "Marital_status",
        "Application_mode",
        "Application_order",
        "Course",
        "Daytime_evening_attendance",
        "Previous_qualification",
        "Previous_qualification_grade",
        "Nacionality",
        "Mothers_qualification",
        "Fathers_qualification",
        "Mothers_occupation",
        "Fathers_occupation",
        "Admission_grade",
        "Displaced",
        "Educational_special_needs",
        "Debtor",
        "Tuition_fees_up_to_date",
        "Gender",
        "Scholarship_holder",
        "Age_at_enrollment",
        "International",
        "Curricular_units_1st_sem_without_evaluations",
        "Curricular_units_2nd_sem_without_evaluations",
        "Unemployment_rate",
        "Inflation_rate",
        "GDP",
        "pc1_1",
        "pc1_2",
        "pc1_3",
        "Status"
    ]
    df = df[desired_order]
    
    return df

if __name__ == "__main__":
    df_raw = pd.read_csv("InstitusiPendidikan_raw.csv", sep=';')
    df_processed = data_preprocessing(df_raw)
    df_processed.to_csv("preprocessing/InstitusiPendidikan_preprocessing.csv", index=False)