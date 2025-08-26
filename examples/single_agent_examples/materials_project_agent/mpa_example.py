from ursa.agents import MaterialsProjectAgent

# make sure your MP_API_KEY is set in env or pass it here
agent = MaterialsProjectAgent(max_results=5)
# example query: 5 stable Ga/In semiconductors with 1.2–2.2 eV gaps
query = {"elements": ["Ga", "In"], "band_gap_min": 1.2, "band_gap_max": 2.2}
context = (
    "1) List each material’s band gap and stability.\n"
    "2) Then, for the single lowest-energy-above-hull candidate, "
    "retrieve and explain its crystal structure."
)
print(agent.run(mp_query=query, context=context))
