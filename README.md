# Boozehound Cocktail Recommendation App

[Try out Boozehound on Heroku!](https://lw-boozehound.herokuapp.com/)

This project was a labor of love for me since it combines two of my favorite things: data and cocktails. Ever since my wife signed us up for a drink mixing class a few years ago I've been stirring up various concoctions, both bartender recipes and original creations. My goal was to create a simple app that would let anyone discover new drink recipes based upon their current favorites or just some descriptive words. Recipe books can be fantastic resources, but I often just want something that tastes like *some other drink but different* or *some combination of flavors and a specific spirit* and that's where a table of contents fails. While I never thought of Boozehound as a replacement for my favorite recipe books I was hoping it could serve as an effective alternative when I just don't have the patience to thumb through dozens of recipes to find what I want to make.

![Picture of a book and index cards containing cocktail recipes](img/resources.jpg)

### Data Collection

Because I wanted Boozehound to work with descriptions and not just cocktail and spirit names I knew I would be relying upon **Natural Language Processing** and would need a fair amount of descriptive text with which to work. I also wanted the app to look good so I needed to get my recipes from a resource that also has images for each drink. I started by scraping the well-designed [Liquor.com](https://www.liquor.com), which has a ton of great recipes and excellent photos. Unfortunately the site has extremely inconsistent write-ups on each cocktail: some are worthy of paragraphs and others only a sentence or two. I wanted longer drink descriptions and I found them at [The Spruce Eats](https://www.thespruceeats.com), which I scraped using [a notebook called 'scrape_spruce_eats'](work/scrape_spruce_eats.ipynb). Spruce Eats doesn't have the greatest list of recipes, but I was still able to collect roughly **980** separate drink entries from the site, each with an ingredient list, description, and an image URL.

### Text Pre-Processing

Getting all of my cocktail recipe data from website to Pandas DataFrame didn't quite get me to the modeling phase as I still needed to format my corpus to make it easier to use. I used the **SpaCy** library to lemmatize words and keep only the nouns and adjectives. **SpaCy** is both fast and easy to use, making it an ideal solution. I then used **scikit-learn's TF-IDF implementation** to create a matrix of each recipe's word frequency vector. I chose **TF-IDF** over other vectorizers because it accounts for word count disparities and some of my drink descriptions are twice as long as others.

### Models

