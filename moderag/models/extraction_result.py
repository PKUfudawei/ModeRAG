from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Union


class Entity(BaseModel):
    """Extracted entity from text (LightRAG style)."""
    entity_name: str = Field(..., description="The name of the entity. Use title case for significant words.")
    entity_type: str = Field(
        default="Other",
        description="Entity type based on its nature. Common types include: Person, Creature, Organization, Location, Event, Concept, Method, Content, Data, Artifact, Equipment, Product, Category, NaturalObject. Create a new type if none fit."
    )
    entity_description: str = Field(
        ...,
        description="A concise yet comprehensive description of the entity's attributes and activities, based solely on the input text."
    )


class Relationship(BaseModel):
    """Extracted relationship from text (LightRAG style)."""
    source_entity: str = Field(
        ...,
        description="The name of the source entity. Ensure consistent naming with entity extraction. Use title case."
    )
    target_entity: str = Field(
        ...,
        description="The name of the target entity. Ensure consistent naming with entity extraction. Use title case."
    )
    relationship_keywords: str = Field(
        ...,
        description="One or more high-level keywords summarizing the overarching nature, concepts, or themes of the relationship. Multiple keywords separated by comma. DO NOT use special delimiters."
    )
    relationship_description: str = Field(
        ...,
        description="A concise explanation of the nature of the relationship between the source and target entities, providing a clear rationale for their connection."
    )


class ExtractionResult(BaseModel):
    """Result of entity and relation extraction (LightRAG style)."""
    model_config = ConfigDict(
        validate_assignment=False,
        extra="ignore",
        use_enum_values=True,
    )
    entities: List[Union[Entity, str]] = Field(
        ...,
        description="Extracted entities (can be strings or {entity_name, entity_type, entity_description} objects)"
    )
    relationships: List[Relationship] = Field(
        ...,
        description="Extracted relationships as list of {source_entity, target_entity, relationship_keywords, relationship_description} objects"
    )

    def _get_entity_names(self) -> set:
        """Get set of all entity names."""
        names = set()
        for e in self.entities:
            if isinstance(e, str):
                names.add(e)
            else:
                names.add(e.entity_name)
        return names

    def validate_edges_reference_existing_entities(self) -> bool:
        """Validate that all edge source/target entities exist in entities list.

        Returns:
            True if all edges reference valid entities.
        Raises:
            ValueError if validation fails.
        """
        entity_names = self._get_entity_names()
        for rel in self.relationships:
            if rel.source_entity not in entity_names:
                raise ValueError(f"Edge source '{rel.source_entity}' not found in entities")
            if rel.target_entity not in entity_names:
                raise ValueError(f"Edge target '{rel.target_entity}' not found in entities")
        return True

    def fix_orphan_edges(self) -> None:
        """Add missing entities from edges to entities list.

        This ensures all edge source/target entities exist in the entities list.
        """
        entity_names = self._get_entity_names()
        added_entities = set()

        for rel in self.relationships:
            if rel.source_entity not in entity_names and rel.source_entity not in added_entities:
                self.entities.append(Entity(entity_name=rel.source_entity, entity_type="Other", entity_description=""))
                added_entities.add(rel.source_entity)
            if rel.target_entity not in entity_names and rel.target_entity not in added_entities:
                self.entities.append(Entity(entity_name=rel.target_entity, entity_type="Other", entity_description=""))
                added_entities.add(rel.target_entity)

    def get_nodes(self) -> List[Entity]:
        """Normalize entities to Entity list."""
        nodes = []
        for e in self.entities:
            if isinstance(e, str):
                nodes.append(Entity(entity_name=e, entity_type="Other", entity_description=""))
            else:
                nodes.append(e)
        return nodes

    def get_edges(self) -> List[Relationship]:
        """Get relationships list."""
        return self.relationships