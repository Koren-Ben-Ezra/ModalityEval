import pandas as pd 

def split_tf_if(input_csv: str, output_csv: str) -> None:
    """
    Read a CSV with columns ['filter', 'correct', 'total', 'accuracy'],
    separate rows ending with '_TF.csv' and '_IF.csv',
    and write a new CSV with columns:
    ['TF_filter', 'TF_accuracy', 'IF_filter', 'IF_accuracy'].
    """
    df = pd.read_csv(input_csv)

    tf_df = df[df['filter'].str.endswith('_TF.csv')][['filter', 'accuracy']]
    if_df = df[df['filter'].str.endswith('_IF.csv')][['filter', 'accuracy']]

    tf_df = tf_df.reset_index(drop=True)
    if_df = if_df.reset_index(drop=True)

    combined = pd.concat([tf_df, if_df], axis=1)
    combined.columns = ['TF_filter', 'TF_accuracy', 'IF_filter', 'IF_accuracy']

    combined.to_csv(output_csv, index=False)

if __name__ == '__main__':
    #ask user of name
    name = input("Enter the directory name (e.g., 'shuffle'): ")
    split_tf_if("summary_" + name + ".csv", "./seperate/summary_" + name + "_separate.csv")
