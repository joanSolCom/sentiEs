import string
from nltk.corpus import stopwords
import spacy
from spacy.lang.es.examples import sentences

def clean_text(text, filterStopwords=False, filterPos=None, lang="es"):

	cleanTokens = []
	stopwordList = []
	nlp = None
	
	if lang=="es":
		nlp = spacy.load('es_core_news_sm')
		stopwordList = stopwords.words('spanish')
		doc = nlp(text)

	elif lang =="en":
		stopwordList = stopwords.words('english')
		nlp = spacy.load('en_core_web_sm')
		doc = nlp(text)

	else:
		return None

	for token in doc:
		#ignore usernames, hashtags, urls and numbers
		if not token.text.startswith("@") and not token.like_url and not token.like_email and not token.like_num and not token.text.startswith("#"):
			if filterPos and not filterStopwords:
				if token.pos_ in filterPos:
					cleanTokens.append(token.text)
			
			elif filterStopwords and not filterPos:
				if token.text not in stopwordList:
					cleanTokens.append(token.text)
			
			elif filterStopwords and filterPos:
				if token.text not in stopwordList and token.pos_ in filterPos:
					cleanTokens.append(token.text)

			elif not filterStopwords and not filterPos:
				cleanTokens.append(token.text)


	
	return cleanTokens

def clean_get_lemmas(text):
	nlp = spacy.load('es_core_news_sm')
	cleanTokens = []
	doc = nlp(text)

	for token in doc:
		cleanTokens.append(token.lemma_)

	return cleanTokens