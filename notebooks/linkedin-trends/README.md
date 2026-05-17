# Labor Market Analysis in Algeria

To analyze labour market trends in Algeria we analyse two alternative datasets - LinkedIn and World Development Indicators. 

## Data Description

This analysis utilizes three key metrics from LinkedIn's Economic Graph to analyze labor market dynamics in South Asia:

### **LinkedIn Hiring Rate (LHR)**
Hiring Rate measures the proportion of LinkedIn members who report starting new jobs each month, providing a real-time indicator of labor market activity.
**Calculation:**  
  - LHR = (Members who added new employers) / (Total country members)  
  - Indexed to the **2016 average** (a value of **1.0** represents the hiring rate of an average month in 2016)
**Presentation:**  
  - Reported as **year-over-year (YOY) percentage changes**, capturing labor market trends relative to the same period in the previous year.
  - Enables **timely month-to-month comparisons** by normalizing fluctuations in hiring patterns.
**Adjustments:**  
  - Accounts for **profile update lags**, ensuring that hiring trends are not skewed by delays in job status changes on LinkedIn.
  - Normalized to remove seasonal hiring variations and provide a clearer picture of underlying labor market dynamics.

> ⚠️ **Note:**  
> The LinkedIn Hiring Rate (LHR) reflects changes in job transitions among LinkedIn members, **not** changes in total employment.  
> Industries with **higher turnover** may show a **higher LHR** even when total employment is stable, because members frequently switch roles and update profiles.  
> Conversely, industries with **low turnover** may show a **lower LHR**, not because the sector is weak, but because job changes occur less frequently.  
> As a result, LHR should be interpreted primarily as a **signal of labor market dynamism and mobility**, rather than a direct measure of job creation.  
> Differences across industries may reflect **structural differences in mobility**, rather than differences in underlying economic performance.

### 2. Skills Genome
Provides an ordered list of most characteristic skills for any entity (occupation, country, industry). It uses the Frequency-Inverse Document Frequency (TF-IDF) algorithm to identify representative skills. The IF-IDF algorithm is a natural language processing (NLP) algorithm to extract the most 'unique' skills. The algorithm essentially allows for a 'weighting' to be created for unique skills. For example, 'Microsoft Word' is a common skill but not a unique skill to a specific job market. Hence, it will be weighted less based on the number of different places it appears. Further details about the methodology can be found [here](https://documents1.worldbank.org/curated/en/827991542143093021/pdf/World-Bank-Group-LinkedIn-Data-Insights-Jobs-Skills-and-Migration-Trends-Methodology-and-Validation-Results.pdf). 

The current dataset being analyzed is the 'POOLED' Skills Genome which shows skills across the years of 2017 - 2024. 


### 3. Skills Penetration
- Measures the prevalence of skill groups within industries and countries, offering insights into workforce specialization and competitiveness.
- Based on the top 50 skills per country-industry combination, ensuring a robust and representative sample.
- Calculated through three key steps:
  1. **National industry skill penetration calculation** – Determines the share of a given skill group within a specific country-industry combination.
  2. **Global industry benchmark estimation** – Establishes an average penetration level for each industry at a global scale.
  3. **Relative penetration value computation** – Compares the national industry value against the global benchmark.

- **Interpretation of Values:**
  - **Values > 1** indicate an above-global-average penetration for a skill group within a country-industry.
  - **Values < 1** indicate a below-global-average penetration.

- **Scope:**
  - Covers **249 skill categories** derived from LinkedIn's **35,000 recognized skills**.
  - Utilizes a dataset spanning **multiple industries and countries**, enabling cross-country and cross-industry comparisons.

The **relative skill penetration metric** allows for meaningful comparisons across countries while accounting for occupational distribution differences on LinkedIn. 

For example, if the **relative tech skill penetration** for **India’s healthcare industry** is **1.2**, this means that the proportion of tech-related skills in India's healthcare workforce is **120% of the global average**, **holding constant for occupational distributions**. This adjustment ensures that differences are not driven by varying workforce compositions across countries but rather by actual skill prevalence.

### 4. Women Representation

This analysis uses LinkedIn’s gender-disaggregated indicators to examine women’s representation across industries, occupations, and skill groups. These indicators reflect the share of LinkedIn members identifying (or inferred) as women within each labor market segment.

**Data construction:**
- Gender is **self-identified** when explicitly listed on the member profile.
- When not self-identified, gender is **inferred** using a combination of profile pronouns and first-name based inference.
- Members whose gender cannot be reliably classified are **excluded** from gender-based indicators.

> ⚠️ **Note:** These indicators should be interpreted as signals of **relative representation on LinkedIn**, not as comprehensive measures of women’s participation in the national labor market.

### Limitations
- Data represents LinkedIn members only
- Coverage varies by country and industry
- Professional network bias in representation
- Industry classifications follow LinkedIn's taxonomy

### Industry Classification

| Industry                                            | Definition                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Accommodation                                       | This industry includes entities that provide short-term lodging in facilities, such as hotels, motels, and bed-and-breakfast inns. In addition to lodging, they may provide a range of other services to their guests.                                                                                                                                                                                                                                                                               |
| Administrative and Support Services                 | This industry includes entities that perform routine support activities for the day-to-day operations of other organizations, including office administration, hiring and placing of personnel, document preparation and similar clerical services, solicitation, collection, security and surveillance services, cleaning, and waste disposal services.                                                                                                                                             |
| Construction                                        | This industry includes entities that construct buildings or engineer projects (e.g., highways and utility systems) and perform specific activities (e.g., painting and plumbing).                                                                                                                                                                                                                                                                                                                    |
| Education                                           | This industry includes entities that provide instruction or training in a wide variety of subjects from specialized entities, such as schools, colleges, universities, and training centers.                                                                                                                                                                                                                                                                                                         |
| Entertainment Providers                             | This industry includes entities that: (1) produce, promote, or participate in live performances, events, or exhibits intended for the public; (2) preserve and exhibit objects and sites of historical, cultural, or educational interest; and (3) operate facilities or provide services that enable patrons to participate in recreational activities or pursue amusement, hobby, and leisure-time interests.                                                                                      |
| Financial Services                                  | This industry includes entities that make financial transactions (creation, liquidation, or change in ownership of financial assets) and/or that facilitate financial transactions.                                                                                                                                                                                                                                                                                                                  |
| Government Administration                           | This industry includes entities of federal, state, and local government agencies that administer, oversee, and manage public programs; organize and finance public goods and services; and have executive, legislative, or judicial authority over other institutions within a given area. These agencies set policy, create laws, adjudicate civil and criminal legal cases, and provide for public safety and national defense.                                                                    |
| Hospitals and Health Care                           | This industry includes entities that provide health care and health-related social assistance for individuals. It includes entities that provide medical care exclusively, health care and social assistance, and only social assistance. These entities deliver services by trained professional health practitioners or social workers.                                                                                                                                                            |
| Manufacturing                                       | This industry includes entities that use mechanical, physical, or chemical transformation of materials, substances, or components to create new products. Included are entities that assemble component parts of manufactured products.                                                                                                                                                                                                                                                              |
| Oil, Gas, and Mining                                | This industry includes entities that extract naturally occurring mineral solids, such as coal and ores; liquid minerals, such as crude petroleum; and gases, such as natural gas. Included are entities that provide quarrying, well operations, and other preparation customarily performed as a part of mining activity.                                                                                                                                                                           |
| Professional Services                               | This industry includes entities that perform professional, scientific, and technical activities for others, including legal advice and representation; accounting, bookkeeping, and payroll services; architectural, engineering, and specialized design services; computer services; consulting services; research services; advertising services; photographic services; translation and interpretation services; veterinary services; and other professional, scientific, and technical services. |
| Retail                                              | This industry includes entities that retail merchandise generally in small quantities to the general public and provide services incidental to the sale of the merchandise.                                                                                                                                                                                                                                                                                                                          |
| Technology, Information and Media                   | This industry includes entities that produce technology products, such as software and data analytics, and provide the means to transmit or distribute these products. Also included are motion picture and sound recording; traditional broadcasting and broadcasting exclusively over the Internet; telecommunications; data processing; and Web search portals and information services.                                                                                                          |
| Transportation, Logistics, Supply Chain and Storage | This industry includes entities that store and warehouse goods, transport passengers and cargo, provide scenic and sightseeing transportation, and provide support activities related to modes of transportation.                                                                                                                                                                                                                                                                                    |

### Skill Classification 


| Skill Group            | Definition                                                                                                                                                                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Soft Skills            | Non-cognitive skills or personality traits valued in the labor market but not assessed by achievement tests. IQ or achievement tests cannot predict these skills.                                                                                       |
| Business Skills        | Knowledge and skills required to start or operate an enterprise. Examples include Business Management, Project Management, Entrepreneurship.                                                                                                            |
| Tech Skills            | Defined as a range of abilities to use digital devices, communication applications, and networks to access and manage information. They enable people to create and share digital content, communicate and collaborate, and solve problems.             |
| Disruptive Tech Skills | Skills associated with developing new technologies that are expected to impact labor markets in the coming years. Examples include Robotics, Genetic Engineering, and Artificial Intelligence (which can be isolated as a skill group category itself). |
| Green Skills           | Skills clearly associated with "green" occupations, per LinkedIn's Green Analytical Methodology                         | 

## Gender Classification

Gender identity isn’t binary and we recognize that some LinkedIn members identify beyond the traditional gender constructs of “men” and “women.” If not explicitly self-identified, we have inferred the gender of members included in this analysis either by the pronouns used on their LinkedIn profiles, or inferred on the basis of first name. Members whose gender could not be inferred as either man or women were excluded from this analysis.
