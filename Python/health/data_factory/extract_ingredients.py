import csv
import json
import operator

# the task is to convert the ingredients csv to a json file, keeping only the information we care about.
# the json file will be a list of products, containing the name of the product, the "health classification" of the product, and the sanitized ingredient listing
#
# In addition, this script is responsible for generating an ingredient word index file.  This is a single file with a unique word in the ingredient corpus on each line.
# this will provide the numerical mapping for the words in the ingredients for use in our NN data


# ingredients needs to report more than this many times to be included
ingredientFrequencyThreshold = 4

# ingredient names must be longer than this
ingredientStringLengthMininum = 4

# ingredient names must be shorter than this
ingredientStringLengthMaximum = 60

ingredientWordDictionary = {}

with open('ingredients.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, quotechar='"')
	
	allProducts = []
	
	for row in reader:
		food_id = 1
		food_asins = 2
		food_brand = 3
		food_categories = 4
		food_dateupdated = 5
		food_ean = 6
		food_features_key = 7
		food_features_value = 8
		food_manufacturer = 9
		food_manufacturer_number = 10
		food_name = 11
		food_sizes = 12
		food_upc = 13
		food_weight = 14

		thisProduct = {}
		
		# default health value
		thisProduct["health"] = 0
		
		#  clean the product name for sanity
		food_name = row[food_name]
		food_name = ''.join([i if ord(i) < 128 else ' ' for i in food_name])
		food_name = ' '.join(food_name.split())
		thisProduct["name"] = food_name
		
				
		# Note: the ingredients strings vary wildly, so we try to normalize them here.  Most are comma separated values, so let's
		# just trim all whitespace, then replace all commas with space, then each ingredient becomes a word
		
		theIngredients = row[8]
		
		# lower all characters
		theIngredients = theIngredients.lower()
		
		# replace contractions with commas
		theIngredients = theIngredients.replace(" with ", ",")
		theIngredients = theIngredients.replace(" & ", ",")
		theIngredients = theIngredients.replace(" and ", ",")
		theIngredients = theIngredients.replace("and ", ",")
		theIngredients = theIngredients.replace(" and", ",")
		theIngredients = theIngredients.replace(" or ", ",")
		theIngredients = theIngredients.replace(" and/or ", ",")
		
		# remove all white spaces
		theIngredients = "".join(theIngredients.split())
		
		# remove stray characters
		theIngredients = theIngredients.replace("[", " ")
		theIngredients = theIngredients.replace("]", " ")
		theIngredients = theIngredients.replace("(", " ")
		theIngredients = theIngredients.replace(")", " ")
		theIngredients = theIngredients.replace(",", " ")
		theIngredients = theIngredients.replace("/", " ")
		theIngredients = theIngredients.replace("--->", " ")
		theIngredients = theIngredients.replace("\\", " ")
		theIngredients = theIngredients.replace(":", " ")
		theIngredients = theIngredients.replace(";", " ")
		theIngredients = theIngredients.replace("!", " ")
		theIngredients = theIngredients.replace(".", " ")
		
		# remove non ascii characters
		theIngredients = ''.join([i if ord(i) < 128 else ' ' for i in theIngredients])
		
		theIngredients = ' '.join(theIngredients.split())
		
		thisProduct["ingredients"] = theIngredients
		
		allProducts.append(thisProduct)
		
		# store each ingredient word
		for word in theIngredients.split():
			if word in ingredientWordDictionary:
				ingredientWordDictionary[word] = ingredientWordDictionary[word] + 1
			else:
				ingredientWordDictionary[word] = 1
		
	
	# export the products json
	#print(json.dumps(allProducts))
	text_file = open("product.json", "w")
	text_file.write(json.dumps(allProducts, sort_keys=True, indent=4, separators=(',', ': ')))
	text_file.close()
	
	# export the ingredients word list
	with open("words.txt", "w") as f:
        
		# just for giggles
		sorted_words = sorted(ingredientWordDictionary.items(), key=operator.itemgetter(1))
		
		for key,value in ingredientWordDictionary.iteritems():
			if value > ingredientFrequencyThreshold and len(key) > ingredientStringLengthMininum and len(key) < ingredientStringLengthMaximum:
				print >>f, "{}".format(key)