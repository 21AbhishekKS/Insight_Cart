from django.db import models

class Product(models.Model):
    source = models.CharField(max_length=100)  # eBay, Amazon, Snapdeal, Ajio
    title = models.CharField(max_length=255)
    price = models.CharField(max_length=50)
    rating = models.CharField(max_length=50, null=True, blank=True)
    link = models.URLField()
    date_scraped = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.title} - {self.source}"

    
class AmazonProduct(models.Model):
    title = models.CharField(max_length=255)
    price = models.CharField(max_length=50)
    rating = models.CharField(max_length=50, null=True, blank=True)
    link = models.URLField()
    date_scraped = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.title} - Amazon"

class eBayProduct(models.Model):
    title = models.CharField(max_length=255)
    price = models.CharField(max_length=50)
    rating = models.CharField(max_length=50, null=True, blank=True)
    link = models.URLField()
    date_scraped = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.title} - eBay"

class SnapdealProduct(models.Model):
    title = models.CharField(max_length=255)
    price = models.CharField(max_length=50)
    rating = models.CharField(max_length=50, null=True, blank=True)
    link = models.URLField()
    date_scraped = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.title} - Snapdeal"

class AjioProduct(models.Model):
    title = models.CharField(max_length=255)
    price = models.CharField(max_length=50)
    rating = models.CharField(max_length=50, null=True, blank=True)
    link = models.URLField()
    date_scraped = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.title} - Ajio"
