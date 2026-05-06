group_analysis
================

# First Research Question

One of the main things we wanted to look at is how health inspection
scores change for the same type of cuisine across different boroughs or
neighborhoods. For example, there are Greek restaurants in both
Manhattaan and Queens. Assuming the distribution of quality is similar
in both boroughs, do we expect to see a difference in health score
ratings?

## Applying Hamza’s Notes

Our dataset currently contains 90 different types of cuisines; however,
many of them can be grouped together. For example `Chinese` and
`Japanese` can be grouped together to simply be `East Asian`, especially
because there’s another group called `Chinese\Japanese`.

We can also drop some restaurants from our dataset for this research
question, as there are some restaurants who are classified as `""` and
`Not Specified / Other`. After grouping into super-groups, we now have
10 types of cuisines:

1.  American
2.  East Asian
3.  South & Southeast Asian
4.  Latin American
5.  European
6.  Mediterranean
7.  Middle Eastern
8.  African
9.  Specific Dishes (Restaurants that only do Sandwiches, Bagels,
    Crepes, etc.)
10. Dietary Specific (Vegan, Gluten-Free, Vegetarian restaurants)

Now, the data looks something like:

``` r
head(df %>% dplyr::select(score, borough, cuisine, hypercategory))
```

    ##   score   borough   cuisine  hypercategory
    ## 1    23 Manhattan   Chinese     East Asian
    ## 2    12  Brooklyn Caribbean Latin American
    ## 3    51 Manhattan   Chinese     East Asian
    ## 4     0  Brooklyn   Chicken      Dish-Type
    ## 5    12 Manhattan  Japanese     East Asian
    ## 6    13     Bronx   Chinese     East Asian

Now for an initial exploratory analysis with graphs. Important to keep
in mind that lower scores are better.

![](group_analysis_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->
Scores on or below the green line are restaurants who earned a `A`
grade. Restaurants who scored above the green line or under the yellow
line are those who scored a `B` grade. Any scores above the yellow
represent a `C` grade.

It may help to also see how the score distribution by cuisine. For a
clearer picture, I will take a random sample.

![](group_analysis_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

From a quick glance, there seems to be a variety of restaurant types in
each grade bucket in our sample. Diving into each cuisine type
specifically,

![](group_analysis_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

Time for regression:

Across all boroughs, do the restaurant of the same cuisine type have the
same expected score?

``` r
fit1 <- lm(score ~ factor(borough) * factor(hypercategory),
           data = df)
```

Bronx is the baseline borough and African is the baseline hypercategory.

``` r
co <- coef(fit1)
se <- sqrt(diag(vcov(fit1)))

keep <- c("factor(borough)Brooklyn",
          "factor(borough)Manhattan",
          "factor(borough)Queens",
          "factor(borough)Staten Island")

arm::coefplot(co[keep], sds = se[keep],
              col.pts = "blue")
```

![](group_analysis_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

Going to ignore other coefficient plots for now, becuase I don’t know
how to work it. But, I think we can do a block design ANOVA?

``` r
sample_block_design <- function(df, n_per_cell) {
  cell_counts <- df %>%
    count(hypercategory, borough, name = "n")

  insufficient <- cell_counts %>%
    filter(n < n_per_cell)
  
  if (nrow(insufficient) > 0) {
    msg <- paste0(
      "Not enough observations in the following hypercategory × borough cells:\n",
      paste0(
        "  - (", insufficient$hypercategory, ", ", insufficient$borough, 
        "): n = ", insufficient$n,
        collapse = "\n"
      )
    )
    stop(msg)
  }
  
  df %>%
    group_by(hypercategory, borough) %>%
    slice_sample(n = n_per_cell, replace = FALSE) %>%
    ungroup()
}
```

Ok, so we should probably drop Staten Island and Dietary Style, since
those are the two most sparsely populated categories.

``` r
display_anova_table(df)
```

    ##                          borough
    ## hypercategory             Bronx Brooklyn Manhattan Queens
    ##   African                   202      167       139     31
    ##   American                 1476     4588     10088   3777
    ##   Beverages & Sweets       1366     4565      6933   3904
    ##   Dish-Type                1970     3491      4434   3010
    ##   East Asian               1080     4172      5763   5491
    ##   European                  952     2035      4484   1489
    ##   Latin American           2022     4516      2235   5237
    ##   Mediterranean              50      700      1114    532
    ##   Middle Eastern            113     1511       651    516
    ##   South & Southeast Asian   220     1717      2614   2485

``` r
anova_model <- aov(score ~ hypercategory * borough, data = sampled_df)
```

``` r
summary(anova_model)
```

    ##                         Df Sum Sq Mean Sq F value   Pr(>F)    
    ## hypercategory            9  22395  2488.3   5.781 6.83e-08 ***
    ## borough                  3    546   182.0   0.423   0.7366    
    ## hypercategory:borough   27  17410   644.8   1.498   0.0492 *  
    ## Residuals             1160 499276   430.4                     
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
