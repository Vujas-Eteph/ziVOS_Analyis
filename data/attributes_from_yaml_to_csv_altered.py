import csv
import yaml

# Load YAML data
yaml_file = './attributes/d17-val/Altered_d17-val_attributes.yaml'
yaml_file = './attributes/lvos-val/Unofficial_lvos-val_attributes.yaml'
with open(yaml_file, 'r') as file:
    data = yaml.safe_load(file)

# Define attributes
attributes = data['attributes']

# Create CSV file and write header
csv_file = './attributes/d17-val/Altered_d17-val_attributes.csv'
csv_file = './attributes/lvos-val/Unofficial_lvos-val_attributes.csv'
with open(csv_file, 'w', newline='') as csvfile:
    fieldnames = ['Sequence Name'] + attributes
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Write data rows
    for sequence in data['sequences']:
        row = {'Sequence Name': sequence['name']}
        for attr in attributes:
            if attr in sequence['attributes']:
                row[attr] = 1
            else:
                row[attr] = 0
        writer.writerow(row)

print("CSV file created successfully!")
