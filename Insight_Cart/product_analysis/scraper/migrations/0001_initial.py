# Generated by Django 5.1.2 on 2024-11-03 14:24

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Product',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('source', models.CharField(max_length=100)),
                ('title', models.CharField(max_length=255)),
                ('price', models.CharField(max_length=50)),
                ('rating', models.CharField(blank=True, max_length=50, null=True)),
                ('link', models.URLField()),
                ('date_scraped', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
