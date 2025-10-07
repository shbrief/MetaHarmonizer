def get_schema_engine():
    from .schema_mapping_engine import SchemaMapEngine
    return SchemaMapEngine


def get_ontology_engine():
    from .ontology_mapping_engine import OntoMapEngine
    return OntoMapEngine
