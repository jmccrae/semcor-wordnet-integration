import wn

def build_ssid_index(wordnet):
    senseid2ssid = {}
    for sense in wordnet.senses():
        sense_id = sense.id[5:]
        lemma, key = sense_id.split("__")
        key = key.replace(".", ":")
        senseid2ssid[lemma + "%" + key] = sense.synset().id
    return senseid2ssid

#wordnet = wn.Wordnet("oewn:2025")
#ssid_index = build_ssid_index(wordnet)
#print(ssid_index["cat%1:05:00::"])
