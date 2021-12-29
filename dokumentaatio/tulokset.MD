## tulokset ja analyysi


## Datan valmistelu
Alkuperäinen aineisto sisälsi yhteensä 581012 havaintoa ja 55 muuttujaa. Aineisto osoittautui liian suureksi, jotta riittävän monimutkainen malli olisi ollut mahdollista toteuttaa käytössä olevalla laskentateholla. Aineistosta karsittiin yhteensä kolme pienemmän pään muuttujaa pois. Nämä olisivat kuvanneet Aspen, Douglas-fir ja Cottonwood/Willow metsätyyppejä. Aineistoon jäi vielä tämän jälkeen 551405 havaintoa. Kahdesta ensimmäisestä muuttujasta karsittiin vielä 160000 ja 228160 havaintoa pois. Jäljelle jäi yhteensä 163245 havaintoa, jotka jakautuvat seuraavasti:

1. 51840 Spruce/Fir (kuusimetsä)
2. 55141 Lodgepole Pine (Kontortamänty)
3. 35754 Ponderosa Pine (Keltamänty)
4. 20510 Krummholz

Supistetun aineistoon numeeriset muutujat normalisoitiin ja jaettiin test- ja train-otoksiin. Test-otoksen kooksi muodostui 25 % prosenttia. Selitettävälle muuttujalle tehtiin one-hot enkoodaus ja selitettävät muuttujat normalisoitiin syväoppimista varten.

## Mallin koonti

Malliksi valittiin Sequential(), joka mahdollisti mallin kerroksittaisen kokoamisen. Malliin lisättiin ensin input-layeri ja yksi hidden-layeri, joissa neuroverkon neuronien määräksi valittiin 16 ja aktivaatiofunktioksi valittiin Relu. Output kerroksen aktivaatiofunktioksi valittiin 'softmax', joka tuottaa kategorisen vektorin. Mallin loss-funktio on 'categorical cross-entropy'(ristientropia), jonka tulisi hyvässä mallissa olla mahdollisimman lähellä nollaa. Ristientropia mittaa virhetta: kuinka kaukana metsätyypin ennustettujakauma on todellisestajakaumasta. Mallin tarkkuutta arvioimiseksi metrics arvoksi valitaan 'accuracy', joka mittaa kuinka usein ennustettu luokka vastaa todellista luokkaa. Mallin optimointi-algoritmiksi valittiin Adam.

## tuloksien ja mallin arviointi

Mallia arvioidaan luokittelu raportin perusteella. Precision-arvo mittaa kuinka suuri prosentti oikeaksi ennustetuista luokist oli oikeita luokkia. Recall-arvo mittaa kuinka moni oikeasti ennustetuista luokista osui oikein. Accuracy on keskeisin mallin suoritusta mittaava arvo ja se on suhde oikein ennustettujen luokkien ja kaikkien havaintojen välillä. F1-arvo on painotettu keskiarvo recall ja precision arvoista. Se kuvastaa mallin ennustettuvuutta, jos alkuperäinen aineisto ei ole tasaisesti jakautunut.

Ristietropia = 0.22
Mallin accuracy: 0.92


![raportti](/kuvat/raportti.png)

Lopullisen mallin loss ja accuracy funktiot:

![lopullinen malli](/kuvat/malli.png)


## hyperparametrien säätäminen

Sopivan mallin etsimiseksi mallia koottii kerros kerrokselta, mallin hyperparametrejä säätäen. Ensin lisäämällä oppimisiteraatioiden eli epochs määrää. Seuraavaksi lisättiin batch_sizea, joka määrittää kuinka ison osan datasta neuroverkolle siirretään kerralla opittavaksi. Batch_size näytti olevan merkittävimmin malliin vaikuttanut hyperparametri. Suurempi algoritmin oppimisnopeus 0.001 johti ylisovittamiseen. Ylisovittaminen oli huomattavasti lievempää oppimisnopeudella 0.01.


Ensimmäisessä mallissa oli vain yksi piilotettu kerros 8 neuronilla, oppimisnopeus oli 0.01 ja batch_sizet olivat 64, 128, 256: Malleista batch_size 256, näytti parhaimmalta ja siinä oli selkeästi vähiten ylioppimista.

![yksi_kerros_node8_batch_size_64_128_252.png](/kuvat/yksi_kerros_node8_batch_size_64_128_252.png)

Neuroverkon hidden layerien määrä pidettiin samana, mutta neuronien määrää nostettin 16 ja batch_sizet nostettiin 128, 256, 512. Oppimisnopeus pidettiin samana 0.01. Tässä mallissa batch_size 256 näytti omasta mielestäni parhaimmalta, vaikka mallissa on vieläkin huomattavissa ylisovittamista.

![yksi_kerros_node16_batch_size_128_256_512.png](/kuvat/yksi_kerros_node16_batch_size_128_256_512.png)

Lisäsin malliin kokeilumielessä vielä toisen hidden layerin. Nyt mallissa oli kaksi kerrosta, joista toisessa oli 16 neuronia ja toisessa 32 neuronia. Oppimisnopeus oli edelleen 0.01 ja batch_sizet 128, 256, 512. Tämä malli osoittautui mielestäni jo liian monimutkaiseksi kyseiseen ongelmaan.

![kaksi_kerros_node_16_ja_32_batch_size_128_256_512.png](/kuvat/kaksi_kerros_node_16_ja_32_batch_size_128_256_512.png)

Kahden piilotetun kerroksen mallin ollessa jo liian monimutkainen. Kokeiltiin nostaa yhden piilotetun kerroksen ja 16 neuronin mallin oppimisnopeutta, jos se parantaisi mallia, mutta se johti ylioppimiseen.

![yksi_kerros_node16_batch_size_128_256_512_lr_0.001.png](/kuvat/yksi_kerros_node16_batch_size_128_256_512_lr_0.001.png)
