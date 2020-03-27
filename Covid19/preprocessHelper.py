from countryinfo import CountryInfo
import pycountry
from googletrans import Translator
import pandas as pd
import numpy as np

gdp = pd.read_csv("gdp.csv")
dict = pd.Series(gdp.gdp.values, index=gdp.Country).to_dict()
print(int(dict['Turkey']))


def createBorderDict(countrySet):
    borderDict = {}
    borderSet = set()
    for c in countrySet:
        try:
            borders = []
            country = CountryInfo(c)
            borders_codes = country.borders()
            for b in borders_codes:
                border = pycountry.countries.get(alpha_3=b).name
                borders.append(border)
                borderSet.add(border)
            borderDict[c] = borders
        except:
            borderDict[c] = borders

    return borderDict, borderSet

def getPopulation(country):
    try:
        country = CountryInfo(country)
        popoulation = country.population()
    except:
        popoulation = 10000000
    return popoulation

def addBorderColumns(df, countrySet):
    borderDict, borderSet = createBorderDict(countrySet)
    for b in borderSet:
        df[b] = df.apply(lambda row: isBorder(borderDict, row['Country'], b), axis=1)
    return df

def getGdp(str):
    gdp = pd.read_csv("gdp.csv")
    dict = pd.Series(gdp.gdp.values, index=gdp.Country).to_dict()
    try:
        result = int(dict[str])
    except:
        result = 10000
    return result
def addGdpPerCapitaColumn(df):
    df['gdp'] = df.apply(lambda row: getGdp(row['Country']), axis=1)
    return df

def addPopulationColumn(df):
    df['Population'] = df.apply(lambda row: getPopulation(row['Country']), axis=1)
    return df

def isBorder(borderDict,srcCountry, borderCountry):
    if borderCountry in borderDict[srcCountry]:
        return 1
    else:
        return 0

def translateCountryNames(countrySet):
    newSet = set()
    translator = Translator()
    for c in countrySet:
        result = translator.translate(c, src='tr', dest='en')
        newSet.add(result.text)
    return newSet

def getDay(date):
    arr = date.split("-")
    return arr[2]

def getMonth(date):
    arr = date.split("-")
    return arr[1]

def getYear(date):
    arr = date.split("-")
    return arr[0]

def addDate(df):
    df.insert(0, 'Year', df.apply(lambda row: getYear(row['date']), axis=1))
    df.insert(0, 'Month', df.apply(lambda row: getMonth(row['date']), axis=1))
    df.insert(0, 'Day', df.apply(lambda row: getDay(row['date']), axis=1))
    df.drop(['date'], axis=1, inplace=True)
    return df

def addVirusDay(df):
    countries = df['Country']
    country = df['Country'][0]
    counter = 0
    virusDay = []
    for c in countries:
        if (country == c):
            counter += 1
        else:
            counter = 1
            country = c
        virusDay.append(counter)
    df['VirusDay'] = virusDay
    return df
