from datasets import load_dataset

dataset_id='germeval18'
remote_test_file="https://raw.githubusercontent.com/uds-lsv/GermEval-2018-Data/master/germeval2018.test.txt"
remote_train_file="https://raw.githubusercontent.com/uds-lsv/GermEval-2018-Data/master/germeval2018.training.txt"


def main():
    dataset = load_dataset('csv', data_files={'train': remote_test_file, 'test': remote_train_file})
    print(dataset)
  
if __name__ == '__main__':
  main()