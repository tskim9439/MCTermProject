import csv
import datetime
import glob

class RawDataFormat:
    def __init__(self, user, ap, time_idx):
        self.user = user
        self.ap = ap
        self.time_idx = time_idx

# Modified raw_type_converter to directly use datetime objects
def raw_type_converter(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None)  # Skip header
        processing_list = []
        for line in csv_reader:
            if not line or not line[0].strip() or line[2].strip() == 'NA':
                continue  # Skip empty lines and lines with 'NA' in AP field

            try:
                date_time = datetime.datetime.strptime(line[0], "%Y-%m-%d %H:%M")
                ap = line[2].replace(" Bldg", "").replace("AP", "_").replace(" ", "")
                temp_raw_data = RawDataFormat(user=line[1], ap=ap, time_idx=date_time)
                processing_list.append(temp_raw_data)
            except ValueError as e:
                print(f"Error parsing date: {line[0]} - {e}")
        return processing_list

def concatnating_raw_data_list(roaming_data):
    connection_map = {}
    for data_ in roaming_data:
        if connection_map.get(data_.user) is None:
            connection_map[data_.user] = []
        connection_map[data_.user].append((data_.ap, data_.time_idx))
    return connection_map

# Modified save_sequences_to_csv to use the datetime object for time formatting
def save_sequences_to_csv(sequences, file_path):
    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file)
        for user_id, sequence in sequences.items():
            ap_names_row = [user_id]
            ap_times_row = ['']
            for ap, date_time in sequence:
                ap_names_row.append(ap)
                ap_times_row.append(date_time.strftime("%Y-%m-%d %H:%M"))
            csv_writer.writerow(ap_names_row)
            csv_writer.writerow(ap_times_row)

# Main processing loop for CSV file
csv_file = r'datasets\2014_01.csv'
output_file_path = r'datasets\2014_01_preprocess_with_time.csv'
for file in glob.glob(csv_file):
    processed_list = raw_type_converter(file)
    user_sequences = concatnating_raw_data_list(processed_list)
    save_sequences_to_csv(user_sequences, output_file_path)
