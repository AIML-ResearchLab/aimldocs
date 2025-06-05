## Introduction to SQL

## What is SQL?
SQL, or Structured Query Language, is a language designed to allow both technical and non-technical users to query, manipulate, and transform data from a relational database. And due to its simplicity, SQL databases provide safe and scalable storage for millions of websites and mobile applications.

There are many popular SQL databases including SQLite, MySQL, Postgres, Oracle and Microsoft SQL Server. All of them support the common SQL language standard, which is what this site will be teaching, but each implementation can differ in the additional features and storage types it supports.


<h2 style="color:blue;">SQL Lesson 1: SELECT queries</h2>

To retrieve data from a SQL database, we need to write SELECT statements.

## Exercise
We will be using a database with data about some of Pixar's classic movies for most of our exercises. This first exercise will only involve the Movies table, and the default query below currently shows all the properties of each movie. 

## Table: movies

| id  | title               | director        | year | length_minutes |
|-----|---------------------|------------------|------|----------------|
| 1   | Toy Story           | John Lasseter    | 1995 | 81             |
| 2   | A Bug's Life        | John Lasseter    | 1998 | 95             |
| 3   | Toy Story 2         | John Lasseter    | 1999 | 93             |
| 4   | Monsters, Inc.      | Pete Docter      | 2001 | 92             |
| 5   | Finding Nemo        | Andrew Stanton   | 2003 | 107            |
| 6   | The Incredibles     | Brad Bird        | 2004 | 116            |
| 7   | Cars                | John Lasseter    | 2006 | 117            |
| 8   | Ratatouille         | Brad Bird        | 2007 | 115            |
| 9   | WALL-E              | Andrew Stanton   | 2008 | 104            |
| 10  | Up                  | Pete Docter      | 2009 | 101            |
| 11  | Toy Story 3         | Lee Unkrich      | 2010 | 103            |
| 12  | Cars 2              | John Lasseter    | 2011 | 120            |
| 13  | Brave               | Brenda Chapman   | 2012 | 102            |
| 14  | Monsters University | Dan Scanlon      | 2013 | 110            |

<div style="color:green;">
1. Find the title of each film

```
SELECT title FROM movies;
```

2. Find the director of each film

```
SELECT director FROM movies;
```

3. Find the title and director of each film

```
SELECT title,director FROM movies;
```

4. Find the title and year of each film

```
SELECT title,year FROM movies;
```

5. Find all the information about each film

```
SELECT * FROM movies;
```

</div>

<h2 style="color:blue;">SQL Lesson 2: Queries with constraints</h2>

Now we know how to select for specific columns of data from a table, but if you had a table with a hundred million rows of data, reading through all the rows would be inefficient and perhaps even impossible.

In order to filter certain results from being returned, we need to use a WHERE clause in the query. The clause is applied to each row of data by checking specific column values to determine whether it should be included in the results or not.

**Select query with constraints**

```
SELECT column, another_column, …
FROM mytable
WHERE condition
    AND/OR another_condition
    AND/OR …;
```

| Operator            | Condition                                   | SQL Example                        |
|---------------------|---------------------------------------------|------------------------------------|
| =, !=, <, <=, >, >= | Standard numerical operators                | col_name != 4                      |
| BETWEEN … AND …     | Number is within range of two values        | col_name BETWEEN 1.5 AND 10.5      |
| NOT BETWEEN … AND … | Number is not within range of two values    | col_name NOT BETWEEN 1 AND 10      |
| IN (…)              | Number exists in a list                     | col_name IN (2, 4, 6)              |
| NOT IN (…)          | Number does not exist in a list             | col_name NOT IN (1, 3, 5)          |


## Exercise
<div style="color:green;">
1. Find the movie with a row id of 6

```
SELECT * FROM movies where id=6
```

2. Find the movies released in the years between 2000 and 2010

```
SELECT * FROM movies where year between 2000 and 2010;
```

3. Find the movies not released in the years between 2000 and 2010

```
SELECT * FROM movies where year not between 2000 and 2010;
```

4. Find the first 5 Pixar movies and their release year

```
SELECT title,Year FROM movies where id<=5;
```
</div>

<h2 style="color:blue;">SQL Lesson 3: Queries with constraints</h2>

When writing WHERE clauses with columns containing text data, SQL supports a number of useful operators to do things like case-insensitive string comparison and wildcard pattern matching. We show a few common text-data specific operators below:

| Operator     | Condition                                                                | Example                                               |
|--------------|--------------------------------------------------------------------------|-------------------------------------------------------|
| =            | Case sensitive exact string comparison (single equals)                  | col_name = "abc"                                      |
| != or <>     | Case sensitive exact string inequality comparison                        | col_name != "abcd"                                    |
| LIKE         | Case insensitive exact string comparison                                 | col_name LIKE "ABC"                                   |
| NOT LIKE     | Case insensitive exact string inequality comparison                      | col_name NOT LIKE "ABCD"                              |
| %            | Matches any sequence of characters (used with LIKE/NOT LIKE)            | col_name LIKE "%AT%" <br>(matches "AT", "ATTIC", "CAT", "BATS") |
| _            | Matches a single character (used with LIKE/NOT LIKE)                    | col_name LIKE "AN_" <br>(matches "AND", not "AN")     |
| IN (…)       | String exists in a list                                                  | col_name IN ("A", "B", "C")                            |
| NOT IN (…)   | String does not exist in a list                                          | col_name NOT IN ("D", "E", "F")                        |


## Exercise
<div style="color:green;">

1. Find all the Toy Story movies

```
SELECT title FROM movies where title like 'Toy Story%'
```

2. Find all the movies directed by John Lasseter

```
SELECT title FROM movies where director like 'John Lasseter%'
```

3. Find all the movies (and director) not directed by John Lasseter

```
SELECT title,director FROM movies where director not like 'John Lasseter%'
```

4. Find all the WALL-* movies

```
SELECT * FROM movies where title like 'WALL-%'
```

</div>

<h2 style="color:blue;">SQL Lesson 4: Filtering and sorting Query results</h2>

Even though the data in a database may be unique, the results of any particular query may not be – take our Movies table for example, many different movies can be released the same year. In such cases, SQL provides a convenient way to discard rows that have a duplicate column value by using the **DISTINCT** keyword.

Since the **DISTINCT** keyword will blindly remove duplicate rows, we will learn in a future lesson how to discard duplicates based on specific columns using grouping and the **GROUP BY** clause.

## Ordering results
To help with this, SQL provides a way to sort your results by a given column in ascending or descending order using the **ORDER BY** clause.

When an ORDER BY clause is specified, each row is sorted alpha-numerically based on the specified column's value.

```
SELECT column, another_column, …
FROM mytable
WHERE condition(s)
ORDER BY column ASC/DESC;
```

## Limiting results to a subset
Another clause which is commonly used with the **ORDER BY** clause are the **LIMIT** and **OFFSET** clauses, which are a useful optimization to indicate to the database the subset of the results you care about.
The **LIMIT** will reduce the number of rows to return, and the optional **OFFSET** will specify where to begin counting the number rows from.

```
SELECT column, another_column, …
FROM mytable
WHERE condition(s)
ORDER BY column ASC/DESC
LIMIT num_limit OFFSET num_offset;
```

## Exercise
<div style="color:green;">

1. List all directors of Pixar movies (alphabetically), without duplicates

```
SELECT DISTINCT(director) FROM movies ORDER BY director
```

2. List the last four Pixar movies released (ordered from most recent to least)

```
SELECT title,year FROM movies ORDER BY year desc  LIMIT 4
```

3. List the first five Pixar movies sorted alphabetically

```
SELECT title FROM movies ORDER BY title  LIMIT 5
```

4. List the next five Pixar movies sorted alphabetically

```
SELECT * FROM movies ORDER BY title  LIMIT 5 OFFSET 5
```
</div>


<h2 style="color:blue;">SQL Review: Simple SELECT Queries</h2>

## Table: north_american_cities

| City                 | Country         | Population | Latitude   | Longitude    |
|----------------------|------------------|------------|------------|--------------|
| Guadalajara          | Mexico           | 1,500,800  | 20.659699  | -103.349609  |
| Toronto              | Canada           | 2,795,060  | 43.653226  | -79.383184   |
| Houston              | United States    | 2,195,914  | 29.760427  | -95.369803   |
| New York             | United States    | 8,405,837  | 40.712784  | -74.005941   |
| Philadelphia         | United States    | 1,553,165  | 39.952584  | -75.165222   |
| Havana               | Cuba             | 2,106,146  | 23.054070  | -82.345189   |
| Mexico City          | Mexico           | 8,555,500  | 19.432608  | -99.133208   |
| Phoenix              | United States    | 1,513,367  | 33.448377  | -112.074037  |
| Los Angeles          | United States    | 3,884,307  | 34.052234  | -118.243685  |
| Ecatepec de Morelos  | Mexico           | 1,742,000  | 19.601841  | -99.050674   |
| Montreal             | Canada           | 1,717,767  | 45.501689  | -73.567256   |
| Chicago              | United States    | 2,718,782  | 41.878114  | -87.629798   |


## Exercise
<div style="color:green;">

1. List all the Canadian cities and their populations

```
SELECT city,population FROM north_american_cities 
where country = 'Canada';
```

2. Order all the cities in the United States by their latitude from north to south

```
SELECT city, latitude 
FROM north_american_cities 
WHERE country = 'United States' 
ORDER BY latitude DESC;
```

3. List all the cities west of Chicago, ordered from west to east

```
SELECT city, longitude
FROM north_american_cities
WHERE longitude < -87.629798
ORDER BY longitude ASC;
```

4. List the two largest cities in Mexico (by population)

```
SELECT city, population
FROM north_american_cities
WHERE country = 'Mexico'
ORDER BY population DESC
LIMIT 2;
```

5. List the third and fourth largest cities (by population) in the United States and their population

```
SELECT city, population
FROM north_american_cities
WHERE country = 'United States'
ORDER BY population DESC
LIMIT 2 OFFSET 2;
```
</div>


<h2 style="color:blue;">SQL Lesson 6: Multi-table queries with JOINs</h2>

## Database normalization
Database normalization is useful because it minimizes duplicate data in any single table, and allows for data in the database to grow independently of each other (ie. Types of car engines can grow independent of each type of car). As a trade-off, queries get slightly more complex since they have to be able to find data from different parts of the database, and performance issues can arise when working with many large tables.

In order to answer questions about an entity that has data spanning multiple tables in a normalized database, we need to learn how to write a query that can combine all that data and pull out exactly the information we need.

## Multi-table queries with JOINs
Tables that share information about a single entity need to have a primary key that identifies that entity uniquely across the database. One common primary key type is an auto-incrementing integer (because they are space efficient), but it can also be a string, hashed value, so long as it is unique.

Using the JOIN clause in a query, we can combine row data across two separate tables using this unique key. The first of the joins that we will introduce is the INNER JOIN.

```
SELECT column, another_table_column, …
FROM mytable
INNER JOIN another_table 
    ON mytable.id = another_table.id
WHERE condition(s)
ORDER BY column, … ASC/DESC
LIMIT num_limit OFFSET num_offset;
```

The **INNER JOIN** is a process that matches rows from the first table and the second table which have the same key (as defined by the ON constraint) to create a result row with the combined columns from both tables. 

## Table: movies (Read-only)
| id  | title               | director        | year | length_minutes |
|-----|---------------------|------------------|------|-----------------|
| 1   | Toy Story           | John Lasseter    | 1995 | 81              |
| 2   | A Bug's Life        | John Lasseter    | 1998 | 95              |
| 3   | Toy Story 2         | John Lasseter    | 1999 | 93              |
| 4   | Monsters, Inc.      | Pete Docter      | 2001 | 92              |
| 5   | Finding Nemo        | Andrew Stanton   | 2003 | 107             |
| 6   | The Incredibles     | Brad Bird        | 2004 | 116             |
| 7   | Cars                | John Lasseter    | 2006 | 117             |
| 8   | Ratatouille         | Brad Bird        | 2007 | 115             |
| 9   | WALL-E              | Andrew Stanton   | 2008 | 104             |
| 10  | Up                  | Pete Docter      | 2009 | 101             |
| 11  | Toy Story 3         | Lee Unkrich      | 2010 | 103             |
| 12  | Cars 2              | John Lasseter    | 2011 | 120             |
| 13  | Brave               | Brenda Chapman   | 2012 | 102             |
| 14  | Monsters University | Dan Scanlon      | 2013 | 110             |

## Table: boxoffice (Read-only)
| movie_id | rating | domestic_sales | international_sales |
|----------|--------|----------------|---------------------|
| 5        | 8.2    | 380843261      | 555900000           |
| 14       | 7.4    | 268492764      | 475066843           |
| 8        | 8.0    | 206445654      | 417277164           |
| 12       | 6.4    | 191452396      | 368400000           |
| 3        | 7.9    | 245852179      | 239163000           |
| 6        | 8.0    | 261441092      | 370001000           |
| 9        | 8.5    | 223808164      | 297503696           |
| 11       | 8.4    | 415004880      | 648167031           |
| 1        | 8.3    | 191796233      | 170162503           |
| 7        | 7.2    | 244082982      | 217900167           |
| 10       | 8.3    | 293004164      | 438338580           |
| 4        | 8.1    | 289916256      | 272900000           |
| 2        | 7.2    | 162798565      | 200600000           |
| 13       | 7.2    | 237283207      | 301700000           |



## Exercise

<div style="color:green;">

1. Find the domestic and international sales for each movie

```
SELECT 
    movies.title,
    boxoffice.domestic_sales,
    boxoffice.international_sales
FROM 
    movies
JOIN 
    boxoffice
ON 
    movies.id = boxoffice.movie_id;
```

2. Show the sales numbers for each movie that did better internationally rather than domestically

```
SELECT 
    movies.title,
    boxoffice.domestic_sales,
    boxoffice.international_sales
FROM 
    movies
JOIN 
    boxoffice
ON 
    movies.id = boxoffice.movie_id
WHERE 
    boxoffice.international_sales > boxoffice.domestic_sales;
```

3. List all the movies by their ratings in descending order

```
SELECT 
    movies.title,
    boxoffice.rating
FROM 
    movies
JOIN 
    boxoffice
ON 
    movies.id = boxoffice.movie_id
ORDER BY 
    boxoffice.rating DESC;
```

</div>


<h2 style="color:blue;">SQL Lesson 7: OUTER JOINs</h2>

Depending on how you want to analyze the data, the INNER JOIN we used last lesson might not be sufficient because the resulting table only contains data that belongs in both of the tables.

If the two tables have asymmetric data, which can easily happen when data is entered in different stages, then we would have to use a LEFT JOIN, RIGHT JOIN or FULL JOIN instead to ensure that the data you need is not left out of the results.

## Select query with LEFT/RIGHT/FULL JOINs on multiple tables

```
SELECT column, another_column, …
FROM mytable
INNER/LEFT/RIGHT/FULL JOIN another_table 
    ON mytable.id = another_table.matching_id
WHERE condition(s)
ORDER BY column, … ASC/DESC
LIMIT num_limit OFFSET num_offset;
```

Like the INNER JOIN these three new joins have to specify which column to join the data on.
- When joining table A to table B, a LEFT JOIN simply includes rows from A regardless of whether a matching row is found in B.
- The RIGHT JOIN is the same, but reversed, keeping rows in B regardless of whether a match is found in A.
- Finally, a FULL JOIN simply means that rows from both tables are kept, regardless of whether a matching row exists in the other table.

## Table: buildings (Read-only)
| building_name | capacity |
|---------------|----------|
| 1e            | 24       |
| 1w            | 32       |
| 2e            | 16       |
| 2w            | 20       |

## Table: employees (Read-only)
| role     | name       | building | years_employed |
|----------|------------|----------|----------------|
| Engineer | Becky A.   | 1e       | 4              |
| Engineer | Dan B.     | 1e       | 2              |
| Engineer | Sharon F.  | 1e       | 6              |
| Engineer | Dan M.     | 1e       | 4              |
| Engineer | Malcom S.  | 1e       | 1              |
| Artist   | Tylar S.   | 2w       | 2              |
| Artist   | Sherman D. | 2w       | 8              |
| Artist   | Jakob J.   | 2w       | 6              |
| Artist   | Lillia A.  | 2w       | 7              |
| Artist   | Brandon J. | 2w       | 7              |
| Manager  | Scott K.   | 1e       | 9              |
| Manager  | Shirlee M. | 1e       | 3              |
| Manager  | Daria O.   | 2w       | 6              |



## Exercise

<div style="color:green;">

1. Find the list of all buildings that have employees

```
SELECT DISTINCT building FROM employees;
```

2. Find the list of all buildings and their capacity

```
SELECT * FROM buildings;
```

3. List all buildings and the distinct employee roles in each building (including empty buildings)

```
SELECT b.building_name, e.role
FROM buildings b
LEFT JOIN employees e ON b.building_name = e.building
GROUP BY b.building_name, e.role
ORDER BY b.building_name, e.role;
```
</div>

<h2 style="color:blue;">SQL Lesson 8: A short note on NULLs</h2>

It's always good to reduce the possibility of NULL values in databases because they require special attention when constructing queries, constraints (certain functions behave differently with null values) and when processing the results.

An alternative to NULL values in your database is to have data-type appropriate default values, like 0 for numerical data, empty strings for text data, etc. But if your database needs to store incomplete data, then NULL values can be appropriate if the default values will skew later analysis (for example, when taking averages of numerical data).

Sometimes, it's also not possible to avoid NULL values, as we saw in the last lesson when outer-joining two tables with asymmetric data. In these cases, you can test a column for NULL values in a WHERE clause by using either the IS NULL or IS NOT NULL constraint.

## Select query with constraints on NULL values
```
SELECT column, another_column, …
FROM mytable
WHERE column IS/IS NOT NULL
AND/OR another_condition
AND/OR …;
```

## Exercise

<div style="color:green;">

1. Find the name and role of all employees who have not been assigned to a building

```
SELECT name, role
FROM employees
WHERE building IS NULL OR building = '';
```

2. Find the names of the buildings that hold no employees

```
SELECT b.building_name
FROM buildings b
LEFT JOIN employees e ON b.building_name = e.building
WHERE e.building IS NULL;
```

</div>


<h2 style="color:blue;">SQL Lesson 9: Queries with expressions</h2>

In addition to querying and referencing raw column data with SQL, you can also use expressions to write more complex logic on column values in a query. These expressions can use mathematical and string functions along with basic arithmetic to transform values when the query is executed, as shown in this physics example.

## Example query with expressions

```
SELECT particle_speed / 2.0 AS half_particle_speed
FROM physics_data
WHERE ABS(particle_position) * 10.0 > 500;
```

Each database has its own supported set of mathematical, string, and date functions that can be used in a query, which you can find in their own respective docs.

The use of expressions can save time and extra post-processing of the result data, but can also make the query harder to read, so we recommend that when expressions are used in the SELECT part of the query, that they are also given a descriptive alias using the AS keyword.

## Select query with expression aliases

```
SELECT col_expression AS expr_description, …
FROM mytable;
```

In addition to expressions, regular columns and even tables can also have aliases to make them easier to reference in the output and as a part of simplifying more complex queries.

## Example query with both column and table name aliases

```
SELECT column AS better_column_name, …
FROM a_long_widgets_table_name AS mywidgets
INNER JOIN widget_sales
  ON mywidgets.id = widget_sales.widget_id;
```


## Exercise

<div style="color:green;">

1. List all movies and their combined sales in millions of dollars

```
SELECT title, (domestic_sales + international_sales) / 1000000 AS gross_sales_millions
FROM movies
  JOIN boxoffice
    ON movies.id = boxoffice.movie_id;
```

2. List all movies and their ratings in percent

```
SELECT title, rating * 10 AS rating_percent
FROM movies
  JOIN boxoffice
    ON movies.id = boxoffice.movie_id;
```

3. List all movies that were released on even number years

```
SELECT title, year
FROM movies
WHERE year % 2 = 0;
```
</div>


<h2 style="color:blue;">SQL Lesson 10: Queries with aggregates</h2>

In addition to the simple expressions that we introduced last lesson, SQL also supports the use of aggregate expressions (or functions) that allow you to summarize information about a group of rows of data. With the Pixar database that you've been using, aggregate functions can be used to answer questions like, "How many movies has Pixar produced?", or "What is the highest grossing Pixar film each year?".

## Select query with aggregate functions over all rows

```
SELECT AGG_FUNC(column_or_expression) AS aggregate_description, …
FROM mytable
WHERE constraint_expression;
```

Without a specified grouping, each aggregate function is going to run on the whole set of result rows and return a single value. And like normal expressions, giving your aggregate functions an alias ensures that the results will be easier to read and process.

## Common aggregate functions

Here are some common aggregate functions that we are going to use in our examples:

| Function              | Description                                                                                                     |
|-----------------------|-----------------------------------------------------------------------------------------------------------------|
| COUNT(*), COUNT(column) | Counts the number of rows in the group if no column is specified; otherwise counts non-NULL values in the specified column. |
| MIN(column)           | Finds the smallest numerical value in the specified column for all rows in the group.                          |
| MAX(column)           | Finds the largest numerical value in the specified column for all rows in the group.                           |
| AVG(column)           | Finds the average numerical value in the specified column for all rows in the group.                           |
| SUM(column)           | Finds the sum of all numerical values in the specified column for the rows in the group.                        |


## Grouped aggregate functions
In addition to aggregating across all the rows, you can instead apply the aggregate functions to individual groups of data within that group (ie. box office sales for Comedies vs Action movies).
This would then create as many results as there are unique groups defined as by the **GROUP BY** clause.

## Select query with aggregate functions over groups

```
SELECT AGG_FUNC(column_or_expression) AS aggregate_description, …
FROM mytable
WHERE constraint_expression
GROUP BY column;
```

The **GROUP BY** clause works by grouping rows that have the same value in the column specified.

## Select query with HAVING constraint

```
SELECT group_by_column, AGG_FUNC(column_expression) AS aggregate_result_alias, …
FROM mytable
WHERE condition
GROUP BY column
HAVING group_condition;
```

The HAVING clause constraints are written the same way as the WHERE clause constraints, and are applied to the grouped rows. With our examples, this might not seem like a particularly useful construct, but if you imagine data with millions of rows with different properties, being able to apply additional constraints is often necessary to quickly make sense of the data.



## Table: employees

| role     | name       | building | years_employed |
|----------|------------|----------|---------------|
| Engineer | Becky A.   | 1e       | 4             |
| Engineer | Dan B.     | 1e       | 2             |
| Engineer | Sharon F.  | 1e       | 6             |
| Engineer | Dan M.     | 1e       | 4             |
| Engineer | Malcom S.  | 1e       | 1             |
| Artist   | Tylar S.   | 2w       | 2             |
| Artist   | Sherman D. | 2w       | 8             |
| Artist   | Jakob J.   | 2w       | 6             |
| Artist   | Lillia A.  | 2w       | 7             |
| Artist   | Brandon J. | 2w       | 7             |
| Manager  | Scott K.   | 1e       | 9             |
| Manager  | Shirlee M. | 1e       | 3             |
| Manager  | Daria O.   | 2w       | 6             |


## Exercise

<div style="color:green;">

1. Find the longest time that an employee has been at the studio

```
SELECT MAX(years_employed) AS longest_time
FROM employees;
```

2. For each role, find the average number of years employed by employees in that role

```
SELECT role, AVG(years_employed) as Average_years_employed
FROM employees
GROUP BY role;
```

3. Find the total number of employee years worked in each building

```
SELECT building, SUM(years_employed) as Total_years_employed
FROM employees
GROUP BY building;
```

4. Find the number of Artists in the studio (without a HAVING clause) 

```
SELECT role, COUNT(*) as Number_of_artists
FROM employees
WHERE role = "Artist";
```

5. Find the number of Employees of each role in the studio

```
SELECT role, COUNT(*)
FROM employees
GROUP BY role;
```

6. Find the total number of years employed by all Engineers

```
SELECT role, SUM(years_employed)
FROM employees
GROUP BY role
HAVING role = "Engineer";
```
</div>


<h2 style="color:blue;">SQL Lesson 11: Order of execution of a Query</h2>

Now that we have an idea of all the parts of a query, we can now talk about how they all fit together in the context of a complete query.

## Complete SELECT query

```
SELECT DISTINCT column, AGG_FUNC(column_or_expression), …
FROM mytable
    JOIN another_table
      ON mytable.column = another_table.column
    WHERE constraint_expression
    GROUP BY column
    HAVING constraint_expression
    ORDER BY column ASC/DESC
    LIMIT count OFFSET COUNT;
```

## Query order of execution

1. ```FROM and JOINs``
2. ```WHERE```
3. ```GROUP BY```
4. ```HAVING```
5. ```SELECT```
6. ```DISTINCT```
7. ```ORDER BY```
8. ```LIMIT / OFFSET```
 
## Table: movies (Read-only)
| id  | title             | director       | year | length_minutes |
|-----|-------------------|----------------|------|----------------|
| 1   | Toy Story         | John Lasseter  | 1995 | 81             |
| 2   | A Bug's Life      | John Lasseter  | 1998 | 95             |
| 3   | Toy Story 2       | John Lasseter  | 1999 | 93             |
| 4   | Monsters, Inc.    | Pete Docter    | 2001 | 92             |
| 5   | Finding Nemo      | Andrew Stanton | 2003 | 107            |
| 6   | The Incredibles   | Brad Bird      | 2004 | 116            |
| 7   | Cars              | John Lasseter  | 2006 | 117            |
| 8   | Ratatouille       | Brad Bird      | 2007 | 115            |
| 9   | WALL-E            | Andrew Stanton | 2008 | 104            |
| 10  | Up                | Pete Docter    | 2009 | 101            |
| 11  | Toy Story 3       | Lee Unkrich    | 2010 | 103            |
| 12  | Cars 2            | John Lasseter  | 2011 | 120            |
| 13  | Brave             | Brenda Chapman | 2012 | 102            |
| 14  | Monsters University | Dan Scanlon  | 2013 | 110            |


## Table: boxoffice (Read-only)
| movie_id | rating | domestic_sales | international_sales |
|----------|--------|----------------|---------------------|
| 5        | 8.2    | 380,843,261    | 555,900,000         |
| 14       | 7.4    | 268,492,764    | 475,066,843         |
| 8        | 8.0    | 206,445,654    | 417,277,164         |
| 12       | 6.4    | 191,452,396    | 368,400,000         |
| 3        | 7.9    | 245,852,179    | 239,163,000         |
| 6        | 8.0    | 261,441,092    | 370,001,000         |
| 9        | 8.5    | 223,808,164    | 297,503,696         |
| 11       | 8.4    | 415,004,880    | 648,167,031         |
| 1        | 8.3    | 191,796,233    | 170,162,503         |
| 7        | 7.2    | 244,082,982    | 217,900,167         |
| 10       | 8.3    | 293,004,164    | 438,338,580         |
| 4        | 8.1    | 289,916,256    | 272,900,000         |
| 2        | 7.2    | 162,798,565    | 200,600,000         |
| 13       | 7.2    | 237,283,207    | 301,700,000         |



## Exercise

<div style="color:green;">

1. Find the number of movies each director has directed

```
SELECT director, COUNT(id) as Num_movies_directed
FROM movies
GROUP BY director;
```

2. Find the total domestic and international sales that can be attributed to each director

```
SELECT director, SUM(domestic_sales + international_sales) as Cumulative_sales_from_all_movies
FROM movies
    INNER JOIN boxoffice
        ON movies.id = boxoffice.movie_id
GROUP BY director;
```

</div>


<h2 style="color:blue;">SQL Lesson 12: Creating tables</h2>

When you have new entities and relationships to store in your database, you can create a new database table using the **CREATE TABLE** statement.

## Create table statement w/ optional table constraint and default value

```
CREATE TABLE IF NOT EXISTS mytable (
    column DataType TableConstraint DEFAULT default_value,
    another_column DataType TableConstraint DEFAULT default_value,
    …
);
```

## Table data types
Different databases support different data types, but the common types support numeric, string, and other miscellaneous things like dates, booleans, or even binary data. Here are some examples that you might use in real code.

| Data Type                 | Description                                                                                                            |
|---------------------------|------------------------------------------------------------------------------------------------------------------------|
| INTEGER, BOOLEAN          | Store whole integer values like counts or ages. Boolean may be represented as 0 or 1.                                  |
| FLOAT, DOUBLE, REAL       | Store precise numerical data with fractional values; different types indicate different floating point precisions.    |
| CHARACTER(num_chars)      | Fixed-length text data type that stores a specific number of characters; may truncate longer values.                   |
| VARCHAR(num_chars)        | Variable-length text data type with a max character limit; more efficient for large tables than fixed-length types.    |
| TEXT                     | Stores strings and text of varying length, typically without a specified max length.                                   |
| DATE, DATETIME           | Store date and time stamps; useful for time series and event data, but can be complex with timezones.                  |
| BLOB                     | Stores binary large objects (binary data); opaque to database, requiring proper metadata for retrieval.                |


## Table constraints

We aren't going to dive too deep into table constraints in this lesson, but each column can have additional table constraints on it which limit what values can be inserted into that column. This is not a comprehensive list, but will show a few common constraints that you might find useful.

| Constraint        | Description                                                                                                                   |
|-------------------|-------------------------------------------------------------------------------------------------------------------------------|
| PRIMARY KEY       | Values in this column are unique and identify a single row in the table.                                                      |
| AUTOINCREMENT     | Automatically fills and increments integer values with each row insertion (not supported in all databases).                   |
| UNIQUE            | Values in this column must be unique, but unlike PRIMARY KEY, it does not necessarily identify a row.                         |
| NOT NULL          | Values inserted in this column cannot be NULL.                                                                                |
| CHECK (expression)| Validates values based on a condition/expression, e.g., ensuring positive values or specific formats.                         |
| FOREIGN KEY       | Ensures that each value in this column corresponds to a valid value in another table’s column, enforcing referential integrity.|

## Movies table schema

```
CREATE TABLE movies (
    id INTEGER PRIMARY KEY,
    title TEXT,
    director TEXT,
    year INTEGER, 
    length_minutes INTEGER
);
```

## Exercise

<div style="color:green;">

1. Create a new table named Database with the following columns:
   – Name A string (text) describing the name of the database
   – Version A number (floating point) of the latest version of this database
   – Download_count An integer count of the number of times this database was downloaded
   - This table has no constraints.

```
CREATE TABLE Database (
    Name TEXT,
    Version FLOAT,
    Download_count INTEGER
);
```

</div>


<h2 style="color:blue;">SQL Lesson 13: Inserting rows</h2>

What is a Schema?
We previously described a table in a database as a two-dimensional set of rows and columns, with the columns being the properties and the rows being instances of the entity in the table. In SQL, the database schema is what describes the structure of each table, and the datatypes that each column of the table can contain.

## Inserting new data

When inserting data into a database, we need to use an **INSERT** statement, which declares which table to write into, the columns of data that we are filling, and one or more rows of data to insert. In general, each row of data you insert should contain values for every corresponding column in the table. You can insert multiple rows at a time by just listing them sequentially.

## Insert statement with values for all columns

```
INSERT INTO mytable
VALUES (value_or_expr, another_value_or_expr, …),
       (value_or_expr_2, another_value_or_expr_2, …),
       …;
```



## Exercise

<div style="color:green;">

1. Add the studio's new production, Toy Story 4 to the list of movies (you can use any director)

```
INSERT INTO movies VALUES (4, "Toy Story 4", "El Directore", 2015, 90);
```

2. Toy Story 4 has been released to critical acclaim! It had a rating of 8.7, and made 340 million domestically and 270 million internationally. Add the record to the BoxOffice table.

```
INSERT INTO boxoffice VALUES (4, 8.7, 340000000, 270000000);
```

</div>


<h2 style="color:blue;">SQL Lesson 13: Updating rows</h2>

In addition to adding new data, a common task is to update existing data, which can be done using an **UPDATE** statement. Similar to the **INSERT** statement, you have to specify exactly which table, columns, and rows to update. In addition, the data you are updating has to match the data type of the columns in the table schema.

## Update statement with values

```
UPDATE mytable
SET column = value_or_expr, 
    other_column = another_value_or_expr, 
    …
WHERE condition;
```


## Exercise

<div style="color:green;">

1. The director for A Bug's Life is incorrect, it was actually directed by John Lasseter

```
UPDATE movies
SET director = "John Lasseter"
WHERE id = 2;
```

2. The year that Toy Story 2 was released is incorrect, it was actually released in 1999

```
UPDATE movies
SET year = 1999
WHERE id = 3;
```

3. Both the title and director for Toy Story 8 is incorrect! The title should be "Toy Story 3" and it was directed by Lee Unkrich

```
UPDATE movies
SET title = "Toy Story 3", director = "Lee Unkrich"
WHERE id = 11;
```

</div>


<h2 style="color:blue;">SQL Lesson 14: Deleting rows</h2>

When you need to delete data from a table in the database, you can use a **DELETE** statement, which describes the table to act on, and the rows of the table to delete through the **WHERE** clause.

## Delete statement with condition

```
DELETE FROM mytable
WHERE condition;
```

If you decide to leave out the WHERE constraint, then all rows are removed, which is a quick and easy way to clear out a table completely (if intentional).

## Taking extra care
Like the UPDATE statement from last lesson, it's recommended that you run the constraint in a SELECT query first to ensure that you are removing the right rows. Without a proper backup or test database, it is downright easy to irrevocably remove data, so always read your DELETE statements twice and execute once.


## Exercise

<div style="color:green;">

1. This database is getting too big, lets remove all movies that were released before 2005.

```
DELETE FROM movies
where year < 2005;
```

2. Andrew Stanton has also left the studio, so please remove all movies directed by him.

```
DELETE FROM movies
where director = "Andrew Stanton";
```

</div>

<h2 style="color:blue;">SQL Lesson 15: Altering tables</h2>

As your data changes over time, SQL provides a way for you to update your corresponding tables and database schemas by using the **ALTER TABLE** statement to add, remove, or modify columns and table constraints.

## Adding columns
The syntax for adding a new column is similar to the syntax when creating new rows in the **CREATE TABLE** statement. You need to specify the data type of the column along with any potential table constraints and default values to be applied to both existing and new rows. In some databases like MySQL, you can even specify where to insert the new column using the **FIRST** or **AFTER** clauses, though this is not a standard feature.

## Altering table to add new column(s)

```
ALTER TABLE mytable
ADD column DataType OptionalTableConstraint 
    DEFAULT default_value;
```

## Removing columns
Dropping columns is as easy as specifying the column to drop, however, some databases (including SQLite) don't support this feature. Instead you may have to create a new table and migrate the data over.

## Altering table to remove column(s)

```
ALTER TABLE mytable
DROP column_to_be_deleted;
```

## Renaming the table
If you need to rename the table itself, you can also do that using the **RENAME** TO clause of the statement.

## Altering table name

```
ALTER TABLE mytable
RENAME TO new_table_name;
```


## Exercise

<div style="color:green;">

1. Add a column named Aspect_ratio with a FLOAT data type to store the aspect-ratio each movie was released in.

```
ALTER TABLE Movies
  ADD COLUMN Aspect_ratio FLOAT DEFAULT 2.39;
```

2. Add another column named Language with a TEXT data type to store the language that the movie was released in. Ensure that the default for this language is English.

```
ALTER TABLE Movies
  ADD COLUMN Language TEXT DEFAULT "English";
```

</div>


<h2 style="color:blue;">SQL Lesson 16: Dropping tables</h2>

In some rare cases, you may want to remove an entire table including all of its data and metadata, and to do so, you can use the DROP TABLE statement, which differs from the DELETE statement in that it also removes the table schema from the database entirely.

## Drop table statement

```
DROP TABLE IF EXISTS mytable;
```

Like the CREATE TABLE statement, the database may throw an error if the specified table does not exist, and to suppress that error, you can use the IF EXISTS clause.

In addition, if you have another table that is dependent on columns in table you are removing (for example, with a FOREIGN KEY dependency) then you will have to either update all dependent tables first to remove the dependent rows or to remove those tables entirely.


## Exercise

<div style="color:green;">

1. We've sadly reached the end of our lessons, lets clean up by removing the Movies table

```
DROP TABLE Movies;
```

2. And drop the BoxOffice table as well

```
DROP TABLE BoxOffice;
```

</div>