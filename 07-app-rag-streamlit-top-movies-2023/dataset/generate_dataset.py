import requests
from bs4 import BeautifulSoup
import pandas as pd

# fetch and parse IMDB top movies page for 2023
def get_imdb_top_movies_2023():
  
  url = "https://www.imdb.com/list/ls562197817/?sort=user_rating,desc&mode=detail"
  response = requests.get(url)
  movies_data = []

  if response.status_code == 200:
    
    soup = BeautifulSoup(response.content, 'html.parser')
    movie_rows = soup.findAll('div', attrs={'class': 'lister-item mode-detail'})
    
    for row in movie_rows:
      
      # tite, rank, year, rating, runtime, genre, votes
      title = row.h3.a.text
      rank = row.h3.find('span', class_ = 'lister-item-index unbold text-primary').text.strip('.')
      year = row.h3.find('span', class_ = 'lister-item-year text-muted unbold').text.strip('()')
      rating = row.find('span', class_ = 'ipl-rating-star__rating').text.replace('\n', '')
      runtime = row.p.find('span', class_ = 'runtime').text.replace(' min', '')
      genre = row.p.find('span', class_ = 'genre').text.strip()
      votes = row.find_all('span', attrs = {'name': 'nv'}).text
      
      certificate = '-1'
      certificate_found = row.find('span', class_ = 'certificate')
      if certificate_found:
        certificate = row.p.find('span', class_ = 'certificate').text
              
      # metascore not available for all movies
      metascore = '-1'
      metascore_found = row.find('span', class_ = 'metascore')
      if metascore_found:
        metascore = row.find('span', class_ = 'metascore').text.replace(' ', '')
      
      # description
      describe = row.find_all('p')
      description = describe[1].text.replace('\n', '') if len(describe) > 1 else '-1'
      
      movies_data.append({
        'Title': title,
        'Rank': rank,
        'Year': year,
        'Rating': rating,
        'Runtime': runtime,
        'Genre': genre,
        'Certificate': certificate,
        'Metascore': metascore,
        'Votes': votes,
        'Gross': gross,
        'Description': description
      })
    
  else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")

  return movies_data


# save to csv
def write_to_csv(movies_data):
  df = pd.DataFrame(movies_data)
  df.to_csv('./imdb_top_movies_2023.csv', index=False)
  df.head(20) 

# main
def main():
  movies_data = get_imdb_top_movies_2023()
  if movies_data:
    write_to_csv(movies_data)
    print("csv dataset created")
  else:
    print("failed to create csv dataset")

if __name__ == "__main__":
  main()
