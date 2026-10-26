from src.data_loader import df

print("\n\nData Preprocessing")

def data_inspection(df):
    """
        This function displays basic information about the dataset.
    """
    print(f"\n\nDataset shape:\n{df.shape}")
    
    print(f"\n\nData types:\n{df.dtypes}")

    print(f"\n\nStatistical Summary:\n{df.describe()}")


def check_missing_values(df):
    """
    Check for missing values and fill them with the column median.
    """
    column_names = ["Attendence", "Marks", "StudyHours", "Assignments"]

    for column in column_names:
        missing_values_count = df[column].isna().sum()

        print(f"Missing values in {column}: {missing_values_count}")

        if missing_values_count > 0:
            df[column] = df[column].fillna(df[column].median())

    return df

def remove_duplicates(df):
    """
        This function checks if duplicates data are presented on the dataset if it is present then all the duplicated datas are dropped
    """
    duplicated_count = df.duplicated().sum()
    print(f"\n\nThe number of duplicated data is {duplicated_count}.")
    if duplicated_count > 0:
        df.drop_duplicates(inplace = True)
    return df

def validate_data(df):
    """
        This function validates data and returns true for all the value that are valid
    """
    print("\n\nData Validation")

    print("\nAttendence Validation: ",df.Attendence.between(0,100).all())
    print("\nMarks Validation: ",df.Marks.between(0,100).all())
    print("\nStudy Hours Validation: ",df.StudyHours.between(0,24).all())
    print("\nAssignment Validation: ",df.Assignments.between(0,10).all())


data_inspection(df)
check_missing_values(df)
remove_duplicates(df)
validate_data(df)