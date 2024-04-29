import pandas as pd
from unidecode import unidecode
import string

YEAR_HANDCODES = {
    'romance and cigarettes': 2005,
    'slow burn': 2005,
    'red tails': 2012,
    'august osage county': 2013,
    'sin city a dame to kill for': 2014,
    'edge of tomorrow': 2014,
    'the last song': 2010,
    'faster': 2010,
    'valentines day': 2010,
    'killers': 2010,
    'the roommate': 2011,
    'the beaver': 2011,
    'the help': 2011,
    'the eagle': 2011,
    'no strings attached': 2011,
    'hanna': 2011,
    'winnie the pooh': 2011,
    'sanctum': 2011,
    'the tree of life': 2011,
    'abduction': 2011,
    'the rite': 2011,
    'unknown': 2011,
    'paul': 2011,
    'the big year': 2011,
    'red riding hood': 2011,
    'the mechanic': 2011,
    'super 8': 2011,
    'upside down': 2012,
    'priest': 2011,
    'hop': 2011,
    'battle los angeles': 2011,
    'the dilemma': 2011,
    'movie 43': 2013,
    'lol': 2012,
    'quartet': 2012,
    'project x': 2012,
    'chronicle': 2012,
    'stoker': 2013,
    'the possession': 2012,
    'promised land': 2012,
    'vamps': 2012,
    'the apparition': 2012,
    'red lights': 2012,
    'lockout': 2012,
    'ride along': 2014,
    'the raven': 2012,
    'lawless': 2012,
    'the vow': 2012,
    'the three stooges': 2012,
    'the butler': 2013,
    'safe': 2012,
    'this is 40': 2012,
    'parker': 2013, 
    'riddick': 2013,
    'the impossible': 2012,
    'big miracle': 2012,
    'taken 2': 2012,
    'ted': 2012,
    'the equalizer': 2014,
    'the watch': 2012,
    'lincoln': 2012,
    'the dictator': 2012,
    'the hunger games': 2012,
    'mirror mirror': 2012,
    'the campaign': 2012,
    'life of pi': 2012,
    'prometheus': 2012,
    'world war z': 2013,
    'battleship': 2012,
    'the avengers': 2012,
    'john carter': 2012,
    'admission': 2013,
    'evil dead': 2013,
    'phantom': 2013,
    'the last stand': 2013,
    '42': 2013,
    'the host': 2013,
    '2 guns': 2013,
    'pompeii': 2014,
    'gravity': 2013,
    'oblivion': 2013,
    'the great gatsby': 2013,
    'man of steel': 2013,
    'neighbors': 2014,
    'metegol': 2013,
    'son of god': 2014,
    'the lego movie': 2014,
    'pixels': 2015,
    'hercules': 2014,
    'the amazing spiderman 2': 2014,
    'burnt': 2015,
    'little boy': 2015,
    'poltergeist': 2015,
    'aloha': 2015,
    'planes fire and rescue': 2014,
    'spy': 2015,
    'ted 2': 2015,
    'minions': 2015,
    'the forest': 2016,
    'shut in': 2016,
    'lion': 2016,
    'robinson crusoe': 2016,
    'genius': 2016,
    'me before you': 2016,
    'kidnap': 2017,
    'the boss': 2016,
    'collide': 2016,
    'criminal': 2016,
    'the accountant': 2016,
    'snowden': 2016,
    'live by night': 2016,
    'warcraft': 2016,
    'the jungle book': 2016,
    'finding dory': 2016,
    'churchill': 2017,
    'gotti': 2018,
    'wish upon': 2017,
    'wonder': 2017,
    'chips': 2017,
    'fist fight': 2017,
    'rings': 2017,
    'the house': 2017,
    'the commuter': 2018,
    'fifty shades darker': 2017,
    'renegades': 2017,
    'the lego batman movie': 2017,
    'power rangers': 2017,
    'logan': 2017,
    'wonder woman': 2017,
    'the mummy': 2017,
    'guardians of the galaxy vol 2': 2017,
    'book club': 2018,
    'overboard': 2018,
    'death wish': 2018,
    'white boy rick': 2018,
    'den of thieves': 2018,
    'game night': 2018,
    'the sisters brothers': 2018,
    'pater rabbit': 2018,
    'annihilation': 2018,
    'the equalizer 2': 2018,
    'the predator': 2018,
    'tomb raider': 2018,
    'bumblebee': 2018,
    'rampage': 2018,
    'skyscraper': 2018,
    'the meg': 2018,
    'miss bala': 2019,
    'glass': 2019,
    'serenity': 2019,
    'leap year': 2010,
    'scary movie 5': 2013,
    'safe house': 2012,
    'tomorrowland': 2015,
    'peter rabbit': 2018,
    'warrior': 2011,
    'mama': 2013,
}


NAME_HANDCODES = {
    'chang jiang qi hao  cj7': 'cj7',
    'stan helsing a parody': 'stan helsing',
    'san suk si gin': 'shinjuku incident',
    'shi yue wei cheng': 'bodyguards and assassins',
    'les intouchables': 'the intouchables',
    'el laberinto del fauno': 'pans labyrinth',
    'lee daniels the butler': 'the butler',
    'mr popperss penguins': 'mr poppers penguins',
    'doctor seuss the lorax': 'the lorax',
    'jin ling shi san chai': 'the flowers of war',
    'san cheng ji': 'a tale of three cities',
    'savva serdtse voyna': 'savva heart of the warrior',
    'star wars ep vii the force awakens': 'star wars the force awakens',
    'chai dan zhuanjia': 'shock wave',
    'star wars ep viii the last jedi': 'star wars the last jedi',
    'spiderman into the spiderverse 3d': 'spiderman into the spiderverse',
    'walle': 'WALL·E',
    'halloween 2': 'halloween ii',
    'michael jacksons this is it': 'this is it',
    'disneys a christmas carol': 'a christmas carol',
    'nanjing nanjing': 'city of life and death',
    'scary movie v': 'scary movie 5',
    'planes fire and rescue': 'planes fire & rescue',
}

def clean_entry(row):
    row['cleaned_name'] = clean_name(row['movie_name'])

    if row['cleaned_name'] in YEAR_HANDCODES:
        row['production_year'] = YEAR_HANDCODES[row['cleaned_name']]

    return row

def clean_name(name):
    """
    Function to clean movie name
    """
    if not name:
        return None 
    
    # Remove punctuation
    cleaned_name = name.translate(str.maketrans('', '', string.punctuation))

    # Remove accents and asian characters
    cleaned_name = unidecode(cleaned_name)

    # Lowercase
    cleaned_name = cleaned_name.lower()

    if cleaned_name in NAME_HANDCODES:
        cleaned_name = NAME_HANDCODES[cleaned_name]

    return cleaned_name


# Get movie list from OpusData sample
opusdata = pd.read_csv('raw/MovieData.csv')

opusdata = opusdata.apply(clean_entry, axis=1) 
print(opusdata)

# Save cleaned data
opusdata.to_csv('output/opus_cleaned.csv', index=False)