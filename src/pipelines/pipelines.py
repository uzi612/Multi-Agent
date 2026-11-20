from src.agents.agents import build_reader_agent, build_search_agent, writter_chain, critic_chain

def run_research_pipiline(topic : str) -> str:
    state = {}

    # Step 1
    print("\n"+" ="*50)
    print("Step 1 - Search Agent working")
    print("="*50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages" : [("user", f"Find recent, reliable and detailed information about : {topic}")]
    })

    state["search_results"] = search_result['messages'][-1].content
    print("\nSearch Result : ", str(state['search_results']).encode('ascii', 'ignore').decode('ascii')[:1000])

    # Step 2 Reader Agent
    print("\n"+" ="*50)
    print("Step 2 - Reader Agent is scraping top resources ..")
    print("="*50)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages" : [("user",
                        f"Based on the following search results about {topic},"
                        f"Pick the most relevant URL and scrape it for deeper content.\n\n"
                        f"Search Results \n:{state['search_results'][:400]}"        
                       )]
    })

    state['scraped_content'] = reader_result['messages'][-1].content
    print("\nScraped Content : \n",str(state['scraped_content']).encode('ascii','ignore').decode('ascii')[:1000])

    # Step 3 Writer Chain
    print("\n"+" ="*50)
    print("Step 3 - Writer is drafting the report ..")
    print("="*50)

    research_combined = (
        f"Search Results : \n{state['search_results'][:1500]} \n\n"
        f"Detailed Scraped content : \n{state['scraped_content'][:2500]}"
    )

    state['report'] = writter_chain.invoke({
        "topic" : topic,
        "research" : research_combined
    })

    print("\nFinal report :\n",str(state['report']).encode('ascii','ignore').decode('ascii'))

    # Step 4 Critic Report
    print("\n"+" ="*50)
    print("Step 3 - Critic is reviewing the report ..")
    print("="*50)

    state['feedback'] = critic_chain.invoke({
        "report" : state['report']
    })

    print("\nCritic report : \n",str(state['feedback']).encode('ascii','ignore').decode('ascii'))

    return state