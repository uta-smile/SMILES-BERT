import os


def main():
    smi_path = os.path.expanduser('~/fastnn-link/fastnn/data/smis')
    # file_name = 'pm2.smi'
    # file_names = ['train', 'valid', 'test']
    file_names = ['nci.smi', 'qm9_1.smi', 'zincp.smi', 'pm2.smi', 'tox21.smi']
    # file_names = ['tox21.csv']
    # output_file = 'nci.smi'
    # output_file = 'tox21.smi'
    # output_file = 'zincp.smi'
    output_file = 'all.smi'
    # sep = ' '
    sep = ','
    # has_head = True
    has_head = False
    smile_idx = -1
    # with open(os.path.join(smi_path, file_name), 'r') as f_reader, open(
    #         os.path.join(smi_path, output_file), 'w') as f_writer:
    #     for line in f_reader:
    #         if len(line.strip()) > 0:
    #             smile = line.split(sep)[0]
    #             f_writer.write('%s\n' % smile)

    with open(os.path.join(smi_path, output_file), 'w') as f_writer:
        for file in file_names:
            with open(os.path.join(smi_path, file), 'r') as f_reader:
                if has_head:
                    f_reader.readline()
                for line in f_reader:
                    if len(line.strip()) > 0:
                        smile = line.strip().split(sep)[smile_idx]
                        f_writer.write('%s\n' % smile)


if __name__ == '__main__':
    main()
