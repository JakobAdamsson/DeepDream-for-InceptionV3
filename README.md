# DeepDream modell
För att utföra experimentet följdes guiden för algoritm och implementation på tensorflow's egna hemsida: https://www.tensorflow.org/tutorials/generative/deepdream

Det som vi författare gjort är att dokumentera koden noggrant med bra förklaringar samt modifera koden på ett sätt som gör att det passa för detta experiment. Men vi vill belysa det faktum att vi inte skrivit eller hitta på algoritmen på egen hand, utan det fanns på Tensorflow.


Deepdream bygger på gradient ascent där man vill maximera loss funktionen med avseende på inputbilden för att modifera bilden i den riktningen där neuronerna i valda lagret reagerar som starkast.

Syftet med att ge denna implementation är för att underlätta för läsaren och för att slippa implementera det själv. Om läsaren på egen hand vill utforska modellen kan man lätt ändra hyperparametrarna: 
1. STEPS
2. STEPSIZE
3. OCTAVE_SCALE

Där OCTAVE_SCALE är en skalär som gör bilden större och därmed skapar starkare mönster medan ett mindre värde skapar en suddigare och mindre drömlik bild.

## Körning av modell
Börja med att skapa ett virtual environment i den directory du önskar, kör sedan
```
python -m venv myenv
```
För att skapa ett virtual environment. Aktivera detta genom att skriva in följande
```
myenv\Scripts\activate
```

Sist behöver du installera alla paket för att köra modellen, det görs genom
```
pip install -r requirements.txt
```
För att köra modellen behöver du köra för ett lager i taget, alltså antingen lager0, lager5 eller lager10. Då du valt vilket lager du vill köra, anpassar du hyperparametrarna efter önskat begär.
Kör sedan igång programmet och invänta att cellen är klar. Bilden som ses efter att programmet exekverat klart är den muterade bilden där beroende på lager, olika mönster kan ses.

