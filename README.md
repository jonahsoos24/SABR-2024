# SABR-2024

Won both our room and Best Overall
- Tasked with analyzing the value of foul balls
- Used a 3-step modeling process to compare the result to a predicted outcome based on pitch characteristics and game state
- Used XGBoost to create a "Stuff" model including pitch characteristics and comparisons to the previous pitch and primary fastball
- Incorporated Stuff Model to capture pitch quality and game state to predict Delta Run Expectancy using an XGBoost Model
- Added in marginal effects to capture the value of fouling a pitch over swinging and missing or not swinging and the value of the additional pitch a batter faces
- Analyzed our metric on a batter level to find who added the most and least value from foul balls
