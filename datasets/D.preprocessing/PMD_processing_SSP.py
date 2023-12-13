#%%
import csv
import glob


# this version has no data balancing process

class RawDataFormat:
    def __init__(self, date, time, user, source, destination):
        self.date = date
        self.time = time
        self.user = user
        self.source = source
        self.destination = destination

    def __cmp__(self, other):
        if self.user == other.user and self.source == other.source and self.destination == other.destination:
            return 1
        else:
            return 0


def raw_type_converter(file_path):
    # Read CSV file
    file = open(file_path, 'r', encoding='utf-8')
    csv_reader = csv.reader(file)

    processing_list = []
    for line in csv_reader:
        temp = str(line)
        if temp.__contains__('%AUTHMGR-4-UNAUTH_MOVE'):
            temp = temp.replace('[', '')
            temp = temp.replace(']', '')
            temp = temp.replace('\'', '')
            temp = temp.split(',')
            # date,time
            date_time = temp[0].split(' ')
            # user
            temp[3] = temp[3].replace('(slow)', '')
            temp[3] = temp[3].replace('(fast)', '')
            user = temp[3][temp[3].find('(') + 1:temp[3].find(')')]
            # source, destination
            source_destination = temp[3].split('from')
            source_destination = source_destination[1].replace(' ', '').split('to')
            source = source_destination[0].replace('Ca', '')
            destination = source_destination[1].replace('Ca', '')
            # data list collection
            temp_raw_data = RawDataFormat(date_time[0], date_time[1], user, source, destination)
            processing_list.append(temp_raw_data)
            # duplicated data delete, only straight sequential data
            if not len(processing_list) == 1:
                if processing_list[-2].__cmp__(temp_raw_data):
                    processing_list.pop()
        else:
            continue
    file.close()
    return reversed(processing_list)


def connection_sequence_geneator(roaming_data):
    connection_map = {}
    for data_ in roaming_data:
        # history sequence find
        if connection_map.__contains__(data_.user):
            # check sequences can be made
            if connection_map[data_.user][-1][-1] == data_.source:
                connection_map[data_.user][-1] += [data_.destination]
            else:
                connection_map[data_.user].append([data_.source, data_.destination])
        else:
            connection_map[data_.user] = [[data_.source, data_.destination]]

    # print(connection_map)
    return connection_map

cnt = 0
def training_data_formater(result_file, sequenced_path_list, window_size):
    # Write CSV file
    file = open(result_file, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(file)
    global cnt
    for list_name in sequenced_path_list:
        # print(list)
        for path in sequenced_path_list[list_name]:
            # print(path)
            # if list_name == 'b02a.4302.71d2':
            #     continue
            # else:
            if len(path) > window_size:
                # print(path)
                for i in range(0, len(path) - window_size):
                    cnt+=1
                    # print(i)
                    # print(path[i:i + window_size+1])
                    csv_writer.writerow(path[i:i + window_size + 1])

    file.close()


# processed_list = raw_type_converter('./2019_csv_files/08-16-2019.csv')
# for data in processed_list:
#     print(data.__dict__)  # __repr__ can be used alternatively (write body before using it)
#
# sequenced_connection_path_data = connection_sequence_geneator(processed_list)
# for data in sequenced_connection_path_data:
#     print(sequenced_connection_path_data[data])  # __repr__ can be used alternatively (write body before using it)

# sequenced_data_for_training(sequenced_connection_path_data, 3)
import os
total_data_count = 0
window_size = 8
result_file_count = 0
result_file_name = '.csv'
connected_path_cnt = 0
save_path = "/home/gtts/MCTermProject/datasets/00.processed_csv_file_ssp/" + str(window_size) + "sequence/"
os.makedirs(save_path, exist_ok=True)
for csv_file in glob.glob('/home/gtts/MCTermProject/datasets/2019_csv_files/*.csv'):
    print(csv_file)
    processed_list = raw_type_converter(csv_file)
    sequenced_connection_path_data = connection_sequence_geneator(processed_list)
    # print(len(sequenced_connection_path_data))
    connected_path_cnt +=len(sequenced_connection_path_data)
    # print(sequenced_connection_path_data)
    result_file_count += 1
    print(save_path + str(result_file_count) + result_file_name)
    training_data_formater(
       save_path + str(result_file_count) + result_file_name,
        sequenced_connection_path_data, window_size)

print(cnt)
print(connected_path_cnt)
print('file generation complete')

# %%
