import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist= input_data.read_data_sets("data/MNIST/", one_hot=True)
#mnist data setinde işlem yapmayı sağlayan kütüphane.
# Bu kodla mnisti indirerek işlem yapmaya hazır hale getiriyoruz.
#labeller one hot olarak istendi yani 10 uzunluğunda vektör.

x= tf.placeholder(tf.float32, [None, 784]) # vektör halince gelen resimler x yer tutucusuna atandı
#vektör uzunluğu olarak 784 belirlendi. None ise sınırlama yok.
#input olarak x aldık. bir de bu labellarda sayıların gerçek değerleri var.

y_true=tf.placeholder(tf.float32, [None,10])# tahmin. 0 dan 9 a kadar 10 tane sayı var

w = tf.Variable(tf.zeros([784,10]))#weight değeridir. tf. variable ile tanımlanan değerler tensorflow tarafından optimize edilir.
#matristeki tüm değerlere 0 atandı. eğitimde optimize edilecekler.w.x yani 784,10 olacak

b=tf.Variable(tf.zeros([10]))#10 tane nöron var çünkü

#y=w.x+b formülüyle gelen değerler genelde logist diye tanımlanır.
logits= tf.matmul(x, w)+b #matmul matris çarpımı fonk

#gelen sonuçlar softmax aktivasyon fonk dan geçirilerek [0,1] arasına sıkıştırılıyor
y=tf.nn.softmax(logits)#10 uzunluğunda vektör var bunun elemanlarından en büyük değer hangisindeyse ona göre
# gelen resmin hangisi olduğunu tahmin edecek

#softmax olasılık gibidir. sigmoid ile karıştırılmamalı. 0 ile 1 arasına öyle sıkışır ki hepsi 0 ile bir arsında olur
# ve toplamları 1 olur mesela 3. nöron 0.9 değer verse model %90 2 olduğunu tahmin ediyor demektir

#tahminin doğrulugunu hesaplayalım
xent=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_true)# iki parametre alır birinci tahmin
# ikinci gerçek değer. softmaxten geçmeden önceki halini veriyoruz ki bu fonk zaten kendi içinde softmax yapıyor.
#ortalama alınır ve çıkan ortalama bizim loss değerimiz olur.

loss=tf.reduce_mean(xent)#daha sonra loss optimize edilmeye çalışılacak

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_true,1))#tahminin doğru mu yanlış mı olduğunu boolean döndürür.
#argmax komutu bizim model ne tahmin etti onu gösterecek yani en aktif düğümün değeri.
#one hot ile aldığımızdan vektörde doğru olan eleman 1 olacak dğerleri 0 olacak

accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#reduce mean ile ortalama alarak modelin ne kadar doğru tahmin yaptığına bakalım
#correct prediction bir boolean old için onu floata dönüştürdük.

#loss değerini bildiğimiz için optimizasyon yapabiliriz.Stokastikgradientdescent kullanacağız
optimize= tf.train.GradientDescentOptimizer(0.5).minimize(loss)#0.5 değeri learning rate değeri.
# en son loss minimize ediliyor

#program bu haliyle çalışmaz oturum açıp tf grafiğine eklememiz lazım.
#optimize etmeden önce global variablesı çalıştırmamız gerekiyor. yapmazsak kodlar çalışmaz
sess=tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 128

def training_step(iterations): #eğitim seti test

    for i in range(iterations):#her resimde farklıı label ve resimleri alarak optimize sağlanacak
        x_batch, y_batch=mnist.train.next_batch(batch_size)
        feed_dict_train={x:x_batch, y_true:y_batch} #x e resim y ye label atadık
        sess.run(optimize, feed_dict=feed_dict_train)

def test_accuracy():    #test seti
    feed_dict_test={x:mnist.test.images, y_true:mnist.test.labels} #x e test resimlerini y ye test labellarını atadık
    acc=sess.run(accuracy, feed_dict=feed_dict_test)
    print('Testing accuracy:', acc)

training_step(2000)
test_accuracy()