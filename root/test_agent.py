from visualization_agent import VisualizationAgent
import pandas as pd

# ׳˜׳¢׳ ׳׳× ׳”׳“׳׳˜׳”
df = pd.read_csv("logs/META-MSFT_log.csv")

# ׳¦׳•׳¨ ׳׳× ׳”׳¡׳•׳›׳
agent = VisualizationAgent(dark_mode=True)

# ׳”׳₪׳§ ׳’׳¨׳£
fig = agent.generate_enhanced_plot(df, pair_name="META-MSFT")
fig.show()
