from keras.models import load_model
import numpy


dataset = numpy.loadtxt("CHDdata.csv", delimiter=",")
datalar = dataset[:, 0:9]
etiketler = dataset[:, 9]
# Eğitilmiş modeli yükle
model = load_model('CHDdataModel.h5')
basarim = model.evaluate(datalar, etiketler)
print("başarım:",basarim[1])

hasta_verileri = [146,10.5,8.29,35.36,1,78,32.73,13.89,53]
ornek_giris = numpy.array([hasta_verileri])

# Model üzerinde tahmin yapın
tahmin = model.predict(ornek_giris)

# Tahmin sonucunu 1 veya 0'a dönüştürün

print("Hasta olma ihtimali : %{:.2%}".format(tahmin[0][0]))