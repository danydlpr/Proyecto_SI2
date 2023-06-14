

1- upload
El archivo debe tener la extensión .xlsx el cual debe tener tipos de datos validos como bool, string (object) e int cada columna necesita su encabezado

2-graficar
Para graficar es necesario tener un archivo cargado

3-entrenar
Para entrenar también es necesario haber cargado correctamente un archivo y enviar la petición con el siguiente formato de ejemplo

	{
	"x":["Area",
			 "genero",
			 "agrupa",
			 "valor",
			 "año",
			 "mes",
			 "indicador"		 
			 ],
	"y": "Categoria",
	"normalizacion": "standard",

	"tecnica": "hold",
	"numero": 20,
	"modelo": "knn"
}

donde x requiere los encabezados del dataset, y es el objetivo, normalizacion puede usar minmax o standard, tecnica usa hold o cross y numero se refiere a el porcentaje de particion del data set o la cantidad de folds que se usaran en el cross validation y modelo se refiere a el modelo de ML que se desea entrenar

4-listar
Listar requiere el nombre del archivo para mostrar los modelos entrenados a partir de este

{
	"nombre":"2023-06-13_22-04-37_datos.xlsx"
}

5-métricas
	Metricas funciona de manera similar a listar pero este arroja toda la información respectiva al entrenamiento de todos los modelos de un archivo en especifico, unicamente solicita el nombre del archivo que debe estar en la base de datos.

{
	"nombre":"2023-06-13_22-04-37_datos.xlsx"
}

6-mejores
Mejores funciona de manera similar a listar. Aquí entregará un top 3 de los modelos entrenados a partir de un dataset particular

{
	"nombre":"2023-06-13_22-04-37_datos.xlsx"
}


7-predecir
predecir cargar un modelo y usa los valores dados para dar una predicción a partir de estos. se requiere un archivo como el del siguiente ejemplo
{
	"modelo":"2023-06-13_22-04-37_datos_arbol",
	"documento":"2023-06-13_23-56-42_datos.xlsx",
	"prediccion":{
		"Area":"Manizales",
             "genero":"femenino",
             "agrupa":"adolescentes",
             "valor":4,
             "año":2022,
             "mes":"Febrero",
             "indicador":"Hurto a personas"


	}
}
