"""Entity and relation extractor using LangChain with json_schema structured output."""
import warnings
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from tqdm import tqdm

from moderag.models.extraction_result import ExtractionResult
from moderag.models.llm import get_llm


class Extractor:
    """Extract entities and relations from text using json_schema structured output.

    Uses LangChain's with_structured_output with json_schema method
    for structured output with Pydantic validation.

    Args:
        llm: LangChain language model instance.
        max_workers: Maximum number of concurrent workers for parallel extraction.
    """

    def __init__(self, llm: Optional[BaseLanguageModel] = None, max_workers: int = 16,
                 raise_on_error: bool = False, max_retries: int = 0):
        # Suppress Pydantic serialization warnings from LangChain internals
        warnings.filterwarnings(
            "ignore",
            message="Pydantic serializer warnings",
            category=UserWarning,
            module="pydantic"
        )
        self.llm = llm or get_llm()
        self.max_workers = max_workers
        # When raise_on_error is True, extraction failures are re-raised to the
        # caller (instead of being swallowed and returning an empty graph doc),
        # so callers can record the real reason. max_retries retries a failing
        # chunk before giving up.
        self.raise_on_error = raise_on_error
        self.max_retries = max_retries
        self._setup_prompt()

    def _setup_prompt(self) -> None:
        """Set up the entity extraction prompt template (LightRAG style)."""
        system_prompt = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.

---Instructions---

1. **Entity Extraction:**
   * **Identification:** Identify clearly defined and meaningful entities in the input text.
   * **Entity Details:** For each identified entity, extract the following information:
     - `entity_name`: The name of the entity. If the entity name is case-insensitive, capitalize the first letter of each significant word (title case). Ensure **consistent naming** across the entire extraction process.
     - `entity_type`: Categorize the entity into an appropriate type based on its nature. Choose from commonly used types such as: Person, Creature, Organization, Location, Event, Concept, Method, Content, Data, Artifact, Equipment, Product, Category, NaturalObject. If none of these types fit, you may create a new type that best describes the entity.
     - `entity_description`: Provide a concise yet comprehensive description of the entity's attributes and activities, based *solely* on the information present in the input text.

2. **Relationship Extraction:**
   * **Identification:** Identify direct, clearly stated, and meaningful relationships between previously extracted entities.
   * **N-ary Relationship Decomposition:** If a single statement describes a relationship involving more than two entities (an N-ary relationship), decompose it into multiple binary (two-entity) relationship pairs for separate description.
     - **Example:** For "Alice, Bob, and Carol collaborated on Project X," extract binary relationships such as "Alice collaborated with Bob," "Alice collaborated with Carol," and "Alice worked on Project X," based on the most reasonable binary interpretations.
   * **Relationship Details:** For each binary relationship, extract the following fields:
     - `source_entity`: The name of the source entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
     - `target_entity`: The name of the target entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
     - `relationship_keywords`: One or more high-level keywords summarizing the overarching nature, concepts, or themes of the relationship. Multiple keywords must be separated by a comma `,`. **DO NOT use special delimiters other than commas.**
     - `relationship_description`: A concise explanation of the nature of the relationship between the source and target entities, providing a clear rationale for their connection.

3. **Relationship Direction & Duplication:**
   * Treat all relationships as **undirected** unless explicitly stated otherwise. Swapping the source and target entities for an undirected relationship does not constitute a new relationship.
   * Avoid outputting duplicate relationships.

4. **Output Order & Prioritization:**
   * Output all extracted entities first, followed by all extracted relationships.
   * Within the list of relationships, prioritize and output those relationships that are **most significant** to the core meaning of the input text first.

5. **Context & Objectivity:**
   * Ensure all entity names and descriptions are written in the **third person**.
   * Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `this paper`, `our company`, `I`, `you`, and `he/she`.

6. **Language & Proper Nouns:**
   * The entire output (entity names, keywords, and descriptions) must be written in the same language as the input text.
   * Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

---Examples---

Example 1:
Input:
```
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
```

Output:
{{
  "entities": [
    {{"entity_name": "Alex", "entity_type": "Person", "entity_description": "A character who experiences frustration and is observant of the dynamics among other characters."}},
    {{"entity_name": "Taylor", "entity_type": "Person", "entity_description": "Portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."}},
    {{"entity_name": "Jordan", "entity_type": "Person", "entity_description": "Shares a commitment to discovery and has a significant interaction with Taylor regarding a device."}},
    {{"entity_name": "Cruz", "entity_type": "Person", "entity_description": "Associated with a vision of control and order, influencing the dynamics among other characters."}},
    {{"entity_name": "The Device", "entity_type": "Artifact", "entity_description": "Central to the story, with potential game-changing implications, and is revered by Taylor."}}
  ],
  "relationships": [
    {{"source_entity": "Alex", "target_entity": "Taylor", "relationship_keywords": "power dynamics, observation", "relationship_description": "Alex observes Taylor's authoritarian behavior and notes changes in Taylor's attitude toward the device."}},
    {{"source_entity": "Alex", "target_entity": "Jordan", "relationship_keywords": "shared goals, rebellion", "relationship_description": "Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."}},
    {{"source_entity": "Taylor", "target_entity": "Jordan", "relationship_keywords": "conflict resolution, mutual respect", "relationship_description": "Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."}},
    {{"source_entity": "Jordan", "target_entity": "Cruz", "relationship_keywords": "ideological conflict, rebellion", "relationship_description": "Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."}},
    {{"source_entity": "Taylor", "target_entity": "The Device", "relationship_keywords": "reverence, technological significance", "relationship_description": "Taylor shows reverence towards the device, indicating its importance and potential impact."}}
  ]
}}

Example 2:
Input:
```
Stock markets faced a sharp downturn today as tech giants saw significant declines, with the global tech index dropping by 3.4% in midday trading. Analysts attribute the selloff to investor concerns over rising interest rates and regulatory uncertainty.

Among the hardest hit, Nexon Technologies saw its stock plummet by 7.8% after reporting lower-than-expected quarterly earnings. In contrast, Omega Energy posted a modest 2.1% gain, driven by rising oil prices.

Meanwhile, commodity markets reflected a mixed sentiment. Gold futures rose by 1.5%, reaching $2,080 per ounce, as investors sought safe-haven assets. Crude oil prices continued their rally, climbing to $87.60 per barrel, supported by supply constraints and strong demand.

Financial experts are closely watching the Federal Reserve's next move, as speculation grows over potential rate hikes. The upcoming policy announcement is expected to influence investor confidence and overall market stability.
```

Output:
{{
  "entities": [
    {{"entity_name": "Global Tech Index", "entity_type": "Category", "entity_description": "Tracks the performance of major technology stocks and experienced a 3.4% decline today."}},
    {{"entity_name": "Nexon Technologies", "entity_type": "Organization", "entity_description": "A tech company that saw its stock decline by 7.8% after disappointing earnings."}},
    {{"entity_name": "Omega Energy", "entity_type": "Organization", "entity_description": "An energy company that gained 2.1% in stock value due to rising oil prices."}},
    {{"entity_name": "Gold Futures", "entity_type": "Product", "entity_description": "Rose by 1.5%, indicating increased investor interest in safe-haven assets."}},
    {{"entity_name": "Crude Oil", "entity_type": "Product", "entity_description": "Prices rose to $87.60 per barrel due to supply constraints and strong demand."}},
    {{"entity_name": "Market Selloff", "entity_type": "Event", "entity_description": "Refers to the significant decline in stock values due to investor concerns over interest rates and regulations."}},
    {{"entity_name": "Federal Reserve Policy Announcement", "entity_type": "Event", "entity_description": "The upcoming policy announcement expected to impact investor confidence and market stability."}}
  ],
  "relationships": [
    {{"source_entity": "Global Tech Index", "target_entity": "Market Selloff", "relationship_keywords": "market performance, investor sentiment", "relationship_description": "The decline in the Global Tech Index is part of the broader market selloff driven by investor concerns."}},
    {{"source_entity": "Nexon Technologies", "target_entity": "Global Tech Index", "relationship_keywords": "company impact, index movement", "relationship_description": "Nexon Technologies' stock decline contributed to the overall drop in the Global Tech Index."}},
    {{"source_entity": "Gold Futures", "target_entity": "Market Selloff", "relationship_keywords": "market reaction, safe-haven investment", "relationship_description": "Gold prices rose as investors sought safe-haven assets during the market selloff."}},
    {{"source_entity": "Federal Reserve Policy Announcement", "target_entity": "Market Selloff", "relationship_keywords": "interest rate impact, financial regulation", "relationship_description": "Speculation over Federal Reserve policy changes contributed to market volatility and investor selloff."}}
  ]
}}

Example 3:
Input:
```
北京是中国的首都，位于华北平原。北京市是中华人民共和国的政治文化中心。
```

Output:
{{
  "entities": [
    {{"entity_name": "北京", "entity_type": "Location", "entity_description": "中国的首都，位于华北平原，是中华人民共和国的政治文化中心。"}},
    {{"entity_name": "中国", "entity_type": "Location", "entity_description": "一个国家，北京是其首都。"}},
    {{"entity_name": "华北平原", "entity_type": "Location", "entity_description": "一个地理区域，北京位于此平原上。"}}
  ],
  "relationships": [
    {{"source_entity": "北京", "target_entity": "中国", "relationship_keywords": "capital, political center", "relationship_description": "北京是中国的首都和政治文化中心。"}},
    {{"source_entity": "北京", "target_entity": "华北平原", "relationship_keywords": "location, geography", "relationship_description": "北京位于华北平原上。"}}
  ]
}}"""

        human_prompt = """---Task---
Extract entities and relationships from the input text in Data to be Processed below.

---Instructions---
1. **Strict Adherence to Format:** Output ONLY valid JSON matching the expected schema. Do not include any introductory or concluding remarks, explanations, or additional text.
2. **Entity Name Consistency:** Ensure all relationship source_entity and target_entity names exactly match entity_name values in the entities list.
3. **Output Language:** Ensure the output language matches the input text language. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

---Data to be Processed---

<Input Text>
```
{text}
```

<Output>
"""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])

    def _build_graph_document(self, result: ExtractionResult, document: Document) -> GraphDocument:
        """Build GraphDocument from ExtractionResult (LightRAG style with metadata)."""
        # Validate and fix orphan edges
        try:
            result.validate_edges_reference_existing_entities()
        except ValueError:
            result.fix_orphan_edges()

        # Build nodes with metadata (LightRAG style)
        nodes_data = result.get_nodes()
        chunk_id = document.metadata.get("chunk_id", "unknown_0")
        file_path = document.metadata.get("source", "unknown_source")

        nodes = []
        for entity in nodes_data:
            node = Node(
                id=entity.entity_name,
                type=entity.entity_type,
                properties={
                    "description": entity.entity_description,
                    "chunk_id": chunk_id,
                    "file_path": file_path,
                }
            )
            nodes.append(node)
        node_map = {node.id: node for node in nodes}

        # Ensure all edge nodes exist
        for rel in result.get_edges():
            if rel.source_entity not in node_map:
                node_map[rel.source_entity] = Node(id=rel.source_entity, type="Other", description="")
            if rel.target_entity not in node_map:
                node_map[rel.target_entity] = Node(id=rel.target_entity, type="Other", description="")

        relationships = []
        for rel in result.get_edges():
            relationship = Relationship(
                source=node_map[rel.source_entity],
                target=node_map[rel.target_entity],
                type="related to",
                properties={
                    "keywords": rel.relationship_keywords,
                    "description": rel.relationship_description,
                    "weight": 1.0,
                    "chunk_id": chunk_id,
                    "file_path": file_path,
                }
            )
            relationships.append(relationship)

        return GraphDocument(
            nodes=nodes,
            relationships=relationships,
            source=document,
        )

    def _invoke_structured(self, text: str) -> ExtractionResult:
        structured_llm = self.llm.with_structured_output(
            ExtractionResult,
            method="json_schema",
            include_raw=False,
            strict=True,
        )
        chain = self.prompt | structured_llm
        return chain.invoke({"text": text})

    async def _ainvoke_structured(self, text: str) -> ExtractionResult:
        structured_llm = self.llm.with_structured_output(
            ExtractionResult,
            method="json_schema",
            include_raw=False,
            strict=True,
        )
        chain = self.prompt | structured_llm
        return await chain.ainvoke({"text": text})

    def _empty_graph(self, document: Document) -> GraphDocument:
        return GraphDocument(nodes=[], relationships=[], source=document)

    def extract(self, document: Document) -> GraphDocument:
        """Extract entities and relations using json_schema structured output.

        Args:
            document: Input LangChain Document.

        Returns:
            LangChain GraphDocument instance.

        Raises:
            The last extraction error if ``raise_on_error`` is set and every
            attempt (1 + max_retries) failed; otherwise returns an empty
            GraphDocument.
        """
        text = document.page_content
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                result = self._invoke_structured(text)
                return self._build_graph_document(result, document)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    print(f"\t[extractor] attempt {attempt + 1}/{self.max_retries} failed, retrying: {e}")

        if self.raise_on_error:
            raise last_error
        print(f"\tError in entity extraction: {last_error}")
        return self._empty_graph(document)

    async def aextract(self, document: Document) -> GraphDocument:
        """Async extract entities and relations.

        Args:
            document: Input LangChain Document.

        Returns:
            LangChain GraphDocument instance.

        Raises:
            The last extraction error if ``raise_on_error`` is set and every
            attempt (1 + max_retries) failed; otherwise returns an empty
            GraphDocument.
        """
        text = document.page_content
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                result = await self._ainvoke_structured(text)
                return self._build_graph_document(result, document)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    print(f"\t[extractor] attempt {attempt + 1}/{self.max_retries} failed, retrying: {e}")

        if self.raise_on_error:
            raise last_error
        print(f"\tError in entity extraction: {last_error}")
        return self._empty_graph(document)

    def extract_batch(
        self,
        documents: List[Document],
        mode: str = "thread"
    ) -> List[GraphDocument]:
        """Batch extract entities and relations from multiple documents.

        Args:
            documents: List of input LangChain Documents.
            mode: Execution mode. Options:
                  - "async": Async concurrent execution (recommended for I/O bound)
                  - "thread": Thread pool concurrent execution
                  - "sync": Sequential execution

        Returns:
            List of LangChain GraphDocument instances.
        """
        if mode not in ['sync', 'thread', 'async']:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: 'sync', 'thread', 'async'")

        if not documents:
            return []

        if mode == "async":
            return asyncio.run(self._extract_batch_async(documents))
        elif mode == "thread":
            return self._extract_batch_thread(documents)
        else:  # mode == "sync"
            return [self.extract(doc) for doc in documents]

    def _extract_batch_thread(self, documents: List[Document]) -> List[GraphDocument]:
        """Thread pool batch extraction with progress bar."""
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.extract, doc) for doc in documents]

            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="Extracting (thread)"
            ):
                results.append(future.result())

        # 按原始顺序返回
        #results.sort(key=lambda r: r.source.metadata.get("chunk_id", 0))
        return results

    async def _extract_batch_async(self, documents: List[Document]) -> List[GraphDocument]:
        """Async batch extraction with concurrency limit and progress bar."""
        semaphore = asyncio.Semaphore(self.max_workers)

        # 创建进度条
        pbar = tqdm(total=len(documents), desc="Extracting (async)", unit="doc")

        async def extract_with_limit(doc: Document) -> GraphDocument:
            result = await self.aextract(doc)
            pbar.update(1)
            return result

        # 并发执行所有任务
        tasks = [extract_with_limit(doc) for doc in documents]
        results = await asyncio.gather(*tasks)

        pbar.close()
        return list(results)


def get_extractor(
    llm: Optional[BaseLanguageModel] = None,
    max_workers: int = 16
) -> Extractor:
    """Get an entity extractor instance.

    Args:
        llm: LangChain language model instance. If None, loads default.
        max_workers: Maximum number of concurrent workers.

    Returns:
        EntityExtractor instance.
    """
    llm = llm or get_llm()
    return Extractor(llm=llm, max_workers=max_workers)


if __name__ == "__main__":
    from langchain_core.documents import Document

    extractor = get_extractor()

    text = "Ebenezer Scrooge is a wealthy but miserly businessman in Victorian London. " \
           "He is visited by the ghost of his former partner Jacob Marley on Christmas Eve."

    doc = Document(page_content=text, metadata={"source": "test"})
    result = extractor.extract(doc)
    print(f"Nodes: {len(result.nodes)}")
    print(f"Relationships: {len(result.relationships)}")
    for node in result.nodes:
        print(f"  - {node.id} ({node.type})")
    for rel in result.relationships:
        print(f"  - {rel.source.id} --{rel.type}--> {rel.target.id}")
