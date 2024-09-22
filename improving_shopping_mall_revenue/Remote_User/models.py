from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)


class predict_revenue(models.Model):

    Tid= models.CharField(max_length=300)
    coupon_no= models.CharField(max_length=300)
    customer_id= models.CharField(max_length=300)
    gender= models.CharField(max_length=300)
    age= models.CharField(max_length=300)
    category= models.CharField(max_length=300)
    quantity= models.CharField(max_length=300)
    price= models.CharField(max_length=300)
    payment_method= models.CharField(max_length=300)
    invoice_date= models.CharField(max_length=300)
    shopping_mall= models.CharField(max_length=300)
    Prediction= models.CharField(max_length=300)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



