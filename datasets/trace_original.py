import json
import csv

# Path to the JSON file
json_file = 'datasets/may26-www.json'

# Path to the CSV file
csv_file = 'trace_original.csv'

# Load the JSON data
with open(json_file) as f:
	data = json.load(f)

# Get the "trace_original" object
trace_original = data.get('trace_original')

columns = [ num for num in range(1, 144) ]

# Check if the object exists
if trace_original:
	# Open the CSV file in write mode
	with open(csv_file, 'w', newline='') as f:
		writer = csv.writer(f)

		# Write the column headers
		writer.writerow(columns)

		# Iterate over the elements in trace_original
		for element in trace_original:
			# Write each element as a row in the CSV file
			writer.writerow(element)

	print('CSV file created successfully.')
else:
	print('The "trace_original" object does not exist in the JSON file.')