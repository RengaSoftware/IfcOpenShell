#!/usr/bin/env python3

# IfcDiff - Compare IFCs
# Copyright (C) 2020, 2021 Dion Moult <dion@thinkmoult.com>
#
# This file is part of IfcDiff.
#
# IfcDiff is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# IfcDiff is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with IfcDiff.  If not, see <http://www.gnu.org/licenses/>.

# This can be packaged with `pyinstaller --onefile --clean --icon=icon.ico ifcdiff.py`

import argparse
import json
import logging
import multiprocessing
import sys
import time
from collections.abc import Iterable
from typing import Any, Literal, Optional, Union

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.classification
import ifcopenshell.util.element
import ifcopenshell.util.placement
import ifcopenshell.util.representation
import ifcopenshell.util.selector
import numpy as np
from deepdiff import DeepDiff
from orderly_set import StableSet

__version__ = version = "0.0.0"


def resolve_mapped_item(mapped_item, processed_ids=None):
    if processed_ids is None:
        processed_ids = []

    if id(mapped_item) in processed_ids:
        return []
    processed_ids.append(id(mapped_item))

    mapping_source = mapped_item.MappingSource
    if not mapping_source:
        return []

    mapped_representation = mapping_source.MappedRepresentation
    if not mapped_representation:
        return []

    result = []

    for item in mapped_representation.Items or []:
        if item.is_a("IfcMappedItem"):
            result.extend(resolve_mapped_item(item, processed_ids))
        else:
            result.append(item)

    return result


def find_mapped_items_in_representation(representation):
    result = []
    for rep in representation.Representations:
        if not rep.is_a("IfcShapeRepresentation"):
            continue

        for item in rep.Items or []:
            if item.is_a("IfcMappedItem"):
                result.extend(resolve_mapped_item(item))
            else:
                result.append(item)

    return result


def get_colors_from_face_set(face_set):
    if not hasattr(face_set, "HasColours") or not face_set.HasColours:
        return []

    result = []

    for colour_map in face_set.HasColours:
        if colour_map.is_a("IfcIndexedColourMap"):
            result.append(colour_map)

    return result


def get_styled_items_from_faceted_brep(brep):
    if not hasattr(brep, "Outer"):
        return []

    outer = brep.Outer
    if not outer:
        return []

    result = []

    for face in outer.CfsFaces or []:
        if not hasattr(face, "StyledByItem"):
            continue
        for styled_item in face.StyledByItem or []:
            if styled_item.is_a("IfcStyledItem"):
                result.append(styled_item)
    return result


def get_material_representations_from_material(material):
    if not hasattr(material, "HasRepresentation"):
        return []

    result = []
    for mat_rep in material.HasRepresentation or []:
        if mat_rep.is_a("IfcMaterialDefinitionRepresentation"):
            result.append(mat_rep)

    return result


def get_material_representations_from_constituent_set(constituent_set):
    result = []

    for con in constituent_set.MaterialConstituents or []:
        if not hasattr(con, "Material"):
            continue

        result.extend(get_material_representations_from_material(con.Material))

    return result


def get_material_representations_from_layer_set_usage(set_usage):
    if not hasattr(set_usage, "ForLayerSet"):
        return []

    layer_set = set_usage.ForLayerSet
    if not layer_set or not hasattr(layer_set, "MaterialLayers"):
        return []

    result = []

    for layer in layer_set.MaterialLayers or []:
        if not hasattr(layer, "Material"):
            continue

        result.extend(get_material_representations_from_material(layer.Material))

    return result


def get_material_representations(element):
    if not hasattr(element, "HasAssociations"):
        return []

    result = []

    for rel in element.HasAssociations or []:
        if not rel.is_a("IfcRelAssociatesMaterial"):
            continue

        rel_material = rel.RelatingMaterial

        if rel_material.is_a("IfcMaterialConstituentSet"):
            result.extend(get_material_representations_from_constituent_set(rel_material))
        elif rel_material.is_a("IfcMaterialLayerSetUsage"):
            result.extend(get_material_representations_from_layer_set_usage(rel_material))
        elif rel_material.is_a("IfcMaterial"):
            result.extend(get_material_representations_from_material(rel_material))
        # IfcMaterialLayerSet, IfcMaterialProfileSet and IfcMaterialProfileSetUsage
        # are omitted: Renga IFC export does not use them.

    return result


class Ifc2Dict:
    def __init__(self, identifiable_classes=None):
        self.identifiable_classes = identifiable_classes or ["IfcProduct"]

    def isIdentifiable(self, object):
        if "__dict__" not in dir(object):
            return False
        if "GlobalId" not in object.__dict__:
            return False
        for c in self.identifiable_classes:
            if object.is_a(c):
                return True
        return False

    def __call__(self, object, foldIdentifiable=False):
        isSimpleValue = lambda object: object is None or isinstance(object, (str, int, float))
        if isSimpleValue(object):
            return object

        if foldIdentifiable and self.isIdentifiable(object):
            return "<{}::{}>".format(object.is_a(), object.GlobalId)

        hasAttributes = lambda object: "__dict__" in dir(object)
        if hasAttributes(object):
            attributes = {key: self.__call__(object.__dict__[key], foldIdentifiable=True) for key in object.__dict__}
            attributes = dict(filter(lambda key_value: key_value[1] is not None, attributes.items()))
            return attributes

        hasContent = lambda object: isinstance(object, Iterable)
        if hasContent(object):
            return [self.__call__(x, foldIdentifiable=True) for x in object]

        return str(object)


RELATIONSHIP_TYPE = Literal[
    "geometry", "attributes", "type", "property", "container", "aggregate", "classification", "inverse"
]


class IfcDiff:
    """Main IfcDiff application

    If you are using IfcDiff as a library, this is the class you should use.

    :param old: IFC file object for the old model
    :param new: IFC file object for the new model
    :param relationships: List of relationships to check. None means that
        attributes and geometry are compared, so changes such as a modified or
        removed PredefinedType are reported. See RELATIONSHIP_TYPE for available
        relationships.
    :param is_shallow: True if you want only the first difference to be listed.
        False if you want all differences to be checked. Choosing False means
        that comparisons will take longer.
    :param filter_elements: An IFC filter query if you only want to compare a
        subset of elements. For example: ``IfcWall`` to only compare walls.
        Takes precedence over identifiable_classes / comparable_classes.
    :param identifiable_classes: IFC classes treated as identity-bearing when
        expanding inverse relationships via Ifc2Dict. Combined with
        comparable_classes to choose which elements to compare.
    :param comparable_classes: IFC classes to scan for comparable elements
        (typically ``IfcRoot``). An entity is compared if it is also one of
        identifiable_classes.
    :param inverse_relationship_classes: Inverse entity classes to DeepDiff
        (typically ``IfcRelationship``). No-op unless "inverse" is in
        relationships.

    Example::

        from ifcdiff import IfcDiff

        ifc_diff = IfcDiff("/path/to/old.ifc", "/path/to/new.ifc", "/path/to/diff.json")
        ifc_diff.diff()
        print(ifc_diff.change_register)
        ifc_diff.export()
    """

    added_elements: set[ifcopenshell.entity_instance]
    deleted_elements: set[ifcopenshell.entity_instance]
    # GlobalIds to changes dictionary.
    change_register: dict[str, dict[str, Any]]

    def __init__(
        self,
        old: ifcopenshell.file,
        new: ifcopenshell.file,
        relationships: Optional[list[RELATIONSHIP_TYPE]] = None,
        is_shallow: bool = True,
        filter_elements: Optional[str] = None,
        identifiable_classes: Optional[list[str]] = None,
        comparable_classes: Optional[list[str]] = None,
        inverse_relationship_classes: Optional[list[str]] = None,
    ):
        self.old = old
        self.new = new
        self.change_register = {}
        self.representation_ids = {}
        self.processed_representation_ids = set()
        self.relationships = relationships or ["attributes", "geometry"]
        self.precision = 1e-4
        self.is_shallow = is_shallow
        self.filter_elements = filter_elements
        self.identifiable_classes = identifiable_classes
        self.comparable_classes = comparable_classes
        self.inverse_relationship_classes = inverse_relationship_classes
        self.equal = False

    def diff(self) -> None:
        logging.disable(logging.CRITICAL)

        self.precision = self.get_precision()
        self.processed_representation_ids = set()

        old_elements = self.collect_element_global_ids(self.old)
        new_elements = self.collect_element_global_ids(self.new)

        print(" - {} item(s) are in the old model".format(len(old_elements)))
        print(" - {} item(s) are in the new model".format(len(new_elements)))

        self.deleted_elements = old_elements - new_elements
        self.added_elements = new_elements - old_elements
        same_elements = new_elements - self.added_elements
        total_same_elements = len(same_elements)

        print(" - {} item(s) were added".format(len(self.added_elements)))
        print(" - {} item(s) were deleted".format(len(self.deleted_elements)))
        print(" - {} item(s) are common to both models".format(total_same_elements))

        total_diffed = 0

        potential_old_changes = []
        potential_new_changes = []

        should_check_attributes = False
        should_check_geometry = False
        should_check_other = False

        for relationship in self.relationships:
            if relationship == "attributes":
                should_check_attributes = True
            elif relationship == "geometry":
                should_check_geometry = True
            else:
                should_check_other = True

        for global_id in same_elements:
            total_diffed += 1
            if total_diffed % 250 == 0:
                print("{}/{} diffed ...".format(total_diffed, total_same_elements), end="\r", flush=True)
            old = self.old.by_guid(global_id)
            new = self.new.by_guid(global_id)
            if should_check_attributes:
                if self.diff_element(old, new) and self.is_shallow:
                    continue
            if should_check_other:
                if self.diff_element_relationships(old, new) and self.is_shallow:
                    continue
            if should_check_geometry:
                self.diff_element_appearance(old, new)
                # Option 1: check everything heuristically using the iterator (seems faster)
                if ifcopenshell.util.representation.get_representation(new, "Model", "Body", "MODEL_VIEW"):
                    potential_old_changes.append(old)
                    potential_new_changes.append(new)
                # Option 2: check first using Python, then fallback to iterator (twice as slow)
                diff = self.diff_element_basic_geometry(old, new)
                if diff:
                   self.change_register.setdefault(new.GlobalId, {}).update({"geometry_changed": True})
                else:
                   potential_old_changes.append(old)
                   potential_new_changes.append(new)

        print(" - {} item(s) had simple changes".format(len(self.change_register.keys())))

        if potential_old_changes:
            print(" - {} item(s) are queued for a detailed geometry check".format(len(potential_old_changes)))
            print("... processing old shapes ...")
            old_shapes = self.summarise_shapes(self.old, potential_old_changes)
            print("... processing new shapes ...")
            new_shapes = self.summarise_shapes(self.new, potential_new_changes)
            print("... comparing shapes ...")
            for global_id, old_shape in old_shapes.items():
                new_shape = new_shapes.get(global_id, None)
                if not new_shape:
                    self._record_geometry_change(global_id, {"mesh_missing_in": "new"})
                    continue
                del new_shapes[global_id]
                diff = DeepDiff(old_shape, new_shape, math_epsilon=1e-5)
                if diff:
                    self._record_geometry_change(global_id, diff)
                    continue

            for global_id in new_shapes.keys():
                self._record_geometry_change(global_id, {"mesh_missing_in": "old"})

        print(" - {} item(s) were changed".format(len(self.change_register.keys())))

        self.equal = (
            len(self.added_elements) == 0 and len(self.deleted_elements) == 0 and len(self.change_register) == 0
        )

        logging.disable(logging.NOTSET)

    def collect_element_global_ids(self, ifc: ifcopenshell.file) -> set[str]:
        if self.filter_elements:
            return set(e.GlobalId for e in ifcopenshell.util.selector.filter_elements(ifc, self.filter_elements))

        if self.identifiable_classes and self.comparable_classes:
            global_ids = set()
            for class_name in self.comparable_classes:
                for entity in ifc.by_type(class_name):
                    if any(entity.is_a(ifc_class) for ifc_class in self.identifiable_classes):
                        global_ids.add(entity.GlobalId)
            return global_ids

        elements = ifc.by_type("IfcElement")
        if ifc.schema == "IFC2X3":
            elements += ifc.by_type("IfcSpatialStructureElement")
        else:
            elements += ifc.by_type("IfcSpatialElement")
        return set(e.GlobalId for e in elements if not e.is_a("IfcFeatureElement"))

    def summarise_shapes(
        self, ifc: ifcopenshell.file, elements: list[ifcopenshell.entity_instance]
    ) -> dict[str, dict[str, Any]]:
        shapes = {}
        products = [e for e in elements if e.is_a("IfcProduct")]
        if not products:
            return shapes
        iterator = ifcopenshell.geom.iterator(
            self.get_settings(ifc), ifc, multiprocessing.cpu_count(), include=products
        )
        valid_file = iterator.initialize()
        while True and valid_file:
            shape = iterator.get()
            element = ifc.by_id(shape.id)
            geometry = shape.geometry
            if geometry.verts:
                shapes[element.GlobalId] = {
                    "total_verts": len(geometry.verts),
                    "sum_verts": sum(geometry.verts),
                    "min_vert": min(geometry.verts),
                    "max_vert": max(geometry.verts),
                    "matrix": tuple(shape.transformation.matrix),
                    "openings": sorted(
                        [o.RelatedOpeningElement.GlobalId for o in getattr(element, "HasOpenings", []) or []]
                    ),
                    "projections": sorted(
                        [o.RelatedFeatureElement.GlobalId for o in getattr(element, "HasProjections", []) or []]
                    ),
                }
            if not iterator.next():
                break
        return shapes

    def get_settings(self, ifc: ifcopenshell.file) -> ifcopenshell.geom.settings:
        settings = ifcopenshell.geom.settings()
        # Are you feeling lucky?
        settings.set("disable-boolean-result", True)
        # Are you feeling very lucky?
        settings.set("disable-opening-subtractions", True)
        # Facetation is to accommodate broken Revit files
        # See https://forums.buildingsmart.org/t/suggestions-on-how-to-improve-clarity-of-representation-context-usage-in-documentation/3663/6?u=moult
        body_contexts = [
            c.id()
            for c in ifc.by_type("IfcGeometricRepresentationSubContext")
            if c.ContextIdentifier in ["Body", "Facetation"]
        ]
        # Ideally, all representations should be in a subcontext, but some BIM programs don't do this correctly
        body_contexts.extend(
            [
                c.id()
                for c in ifc.by_type("IfcGeometricRepresentationContext", include_subtypes=False)
                if c.ContextType == "Model"
            ]
        )
        if body_contexts:
            settings.set("context-ids", body_contexts)
        return settings

    def json_dump_default(self, obj):
        # result of DeepDiff may contain ordered sets
        if isinstance(obj, (StableSet, set, frozenset)):
            return list(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if type(obj).__name__ in ("SetOrdered", "OrderedSet"):
            return list(obj)
        return json.JSONEncoder.default(None, obj)

    def export(self, path: Optional[str] = None) -> None:
        if path is None:
            return
        with open(path, "w", encoding="utf-8") as diff_file:
            json.dump(
                {
                    "added": list(self.added_elements),
                    "deleted": list(self.deleted_elements),
                    "changed": self.change_register,
                },
                diff_file,
                indent=4,
                default=self.json_dump_default,
            )

    def get_precision(self) -> float:
        contexts = [c for c in self.new.by_type("IfcGeometricRepresentationContext") if c.ContextType == "Model"]
        if contexts:
            return contexts[0].Precision or 1e-4
        return 1e-4

    def diff_element(self, old, new):
        diff = DeepDiff(
            [a for a in old if not isinstance(a, (ifcopenshell.entity_instance, tuple))],
            [a for a in new if not isinstance(a, (ifcopenshell.entity_instance, tuple))],
            math_epsilon=self.precision,
            ignore_string_type_changes=True,
            ignore_numeric_type_changes=True,
        )
        if diff and new.GlobalId:
            self.change_register.setdefault(new.GlobalId, {}).update({"attributes_changed": True})
            return True

    def diff_element_relationships(self, old, new):
        if not self.relationships:
            return
        for relationship in self.relationships:
            if relationship == "type":
                old_type = ifcopenshell.util.element.get_type(old)
                new_type = ifcopenshell.util.element.get_type(new)
                if old_type is not None and new_type is not None:
                    if old_type.GlobalId != new_type.GlobalId:
                        self.change_register.setdefault(new.GlobalId, {}).update({"type_changed": True})
                        return True
                elif old_type != new_type:
                    # one of the types is None while the other is not None
                    self.change_register.setdefault(new.GlobalId, {}).update({"type_changed": True})
                    return True
            elif relationship == "property":
                old_psets = ifcopenshell.util.element.get_psets(old)
                new_psets = ifcopenshell.util.element.get_psets(new)
                try:
                    diff = DeepDiff(
                        old_psets,
                        new_psets,
                        math_epsilon=self.precision,
                        ignore_string_type_changes=True,
                        ignore_numeric_type_changes=True,
                        exclude_regex_paths=[r".*id$"],
                    )
                except:
                    diff = True
                if diff and new.GlobalId:
                    self.change_register.setdefault(new.GlobalId, {}).update({"properties_changed": diff})
                    return True
            elif relationship == "container":
                if ifcopenshell.util.element.get_container(old) != ifcopenshell.util.element.get_container(new):
                    self.change_register.setdefault(new.GlobalId, {}).update({"container_changed": True})
                    return True
            elif relationship == "aggregate":
                if ifcopenshell.util.element.get_aggregate(old) != ifcopenshell.util.element.get_aggregate(new):
                    self.change_register.setdefault(new.GlobalId, {}).update({"aggregate_changed": True})
                    return True
            elif relationship == "classification":
                old_id = "ItemReference" if self.old.schema == "IFC2X3" else "Identification"
                new_id = "ItemReference" if self.new.schema == "IFC2X3" else "Identification"
                old_refs = [getattr(r, old_id) for r in ifcopenshell.util.classification.get_references(old)]
                new_refs = [getattr(r, new_id) for r in ifcopenshell.util.classification.get_references(new)]
                if old_refs != new_refs:
                    self.change_register.setdefault(new.GlobalId, {}).update({"classification_changed": True})
                    return True
            elif relationship == "inverse":
                if not self.inverse_relationship_classes:
                    continue

                ifc_expander = Ifc2Dict(self.identifiable_classes)
                old_relationships = [
                    ifc_expander(x)
                    for x in self.old.get_inverse(old)
                    for c in self.inverse_relationship_classes
                    if x.is_a(c)
                ]
                new_relationships = [
                    ifc_expander(x)
                    for x in self.new.get_inverse(new)
                    for c in self.inverse_relationship_classes
                    if x.is_a(c)
                ]

                diff = DeepDiff(
                    old_relationships,
                    new_relationships,
                    significant_digits=6,
                    ignore_string_type_changes=True,
                    ignore_numeric_type_changes=True,
                    ignore_order=True,
                    exclude_regex_paths=[
                        r".*\'id\'.*",
                        r".*\'GlobalId\'.*",
                        r"root.*SelfIntersect$",
                    ],
                )
                if diff:
                    self.change_register.setdefault(new.GlobalId, {}).update({"inverse_relations_changes": diff})
                    return True

    def _record_geometry_change(self, global_id: str, details) -> None:
        entry = self.change_register.setdefault(global_id, {})
        entry["geometry_changed"] = True
        entry.setdefault("geometry_changed_details", details)

    def diff_element_basic_geometry(self, old, new):
        has_placement = lambda e: "ObjectPlacement" in getattr(e, "__dict__", {}) and e.ObjectPlacement is not None
        if has_placement(old) and has_placement(new):
            old_placement = ifcopenshell.util.placement.get_local_placement(old.ObjectPlacement)
            new_placement = ifcopenshell.util.placement.get_local_placement(new.ObjectPlacement)
            if not np.allclose(old_placement[:, 3], new_placement[:, 3], atol=self.precision):
                return True
            if not np.allclose(old_placement[0:3, 0:3], new_placement[0:3, 0:3], atol=1e-2):
                return True
        elif has_placement(old) != has_placement(new):
            return True
        old_openings = sorted([o.RelatedOpeningElement.GlobalId for o in getattr(old, "HasOpenings", []) or []])
        new_openings = sorted([o.RelatedOpeningElement.GlobalId for o in getattr(new, "HasOpenings", []) or []])
        if old_openings != new_openings:
            return True
        old_projections = sorted([o.RelatedFeatureElement.GlobalId for o in getattr(old, "HasProjections", []) or []])
        new_projections = sorted([o.RelatedFeatureElement.GlobalId for o in getattr(new, "HasProjections", []) or []])
        if old_projections != new_projections:
            return True
        # Option 3: check completely using Python with get_info (extremely slow, not worth it)
        old_rep_id = self.get_representation_id(old)
        new_rep_id = self.get_representation_id(new)
        rep_result = self.representation_ids.get(new_rep_id, None)
        if rep_result is not None:
           return rep_result
        if type(old_rep_id) != type(new_rep_id):
           self.representation_ids[new_rep_id] = True
           return True
        if new_rep_id is None:
           return
        result = self.diff_representation(old_rep_id, new_rep_id, old.GlobalId) or False
        self.representation_ids[new_rep_id] = result
        return result

    def diff_representation(self, old_rep_id: int, new_rep_id: int, global_id) -> bool:
        old_rep = self.old.by_id(old_rep_id)
        new_rep = self.new.by_id(new_rep_id)
        if len(old_rep.Items) != len(new_rep.Items):
            return True
        for i, old_item in enumerate(old_rep.Items):
            result = self.diff_representation_item(old_item, new_rep.Items[i], global_id)
            if result is True:
                return True
        return False

    def diff_representation_item(
        self, old_item: ifcopenshell.entity_instance, new_item: ifcopenshell.entity_instance, global_id) -> bool:
        if old_item.is_a() != new_item.is_a():
            return True
        
        diff = DeepDiff(
            old_item.get_info(recursive=True),
            new_item.get_info(recursive=True),
            significant_digits=6,
            ignore_string_type_changes=True,
            ignore_numeric_type_changes=True,
            ignore_order=True,
            exclude_regex_paths=[r".*id']$"],
        )        
        if diff:
            self.change_register.setdefault(global_id, {}).update({"geometry_changed_details" : diff})
            return True
        return False

    def record_deep_diff(self, global_id: str, key: str, old_value, new_value, **kwargs) -> bool:
        settings = {
            "significant_digits": 6,
            "ignore_string_type_changes": True,
            "ignore_numeric_type_changes": True,
            "ignore_order": True,
        }
        settings.update(kwargs)
        diff = DeepDiff(old_value, new_value, **settings)
        if diff and global_id:
            self.change_register.setdefault(global_id, {}).update({key: diff})
            return True
        return False

    def product_representation(self, element: ifcopenshell.entity_instance):
        if "Representation" in getattr(element, "__dict__", {}) and element.Representation is not None:
            return element.Representation
        return None

    def diff_element_appearance(self, old, new) -> bool:
        changed = self.diff_materials(old, new)

        old_representation = self.product_representation(old)
        new_representation = self.product_representation(new)
        if not old_representation or not new_representation:
            return changed

        representation_id = self.get_representation_id(new)
        if representation_id is not None:
            if representation_id in self.processed_representation_ids:
                return changed
            self.processed_representation_ids.add(representation_id)

        global_id = new.GlobalId
        changed = self.diff_presentation_layer_assignment(global_id, old_representation, new_representation) or changed
        changed = self.diff_shape_aspects(global_id, old_representation, new_representation) or changed
        changed = self.diff_styled_items(global_id, old_representation, new_representation) or changed
        changed = self.diff_colors(global_id, old_representation, new_representation) or changed
        return changed

    def diff_colors(self, global_id, old_representation, new_representation) -> bool:
        old_mapped_items = find_mapped_items_in_representation(old_representation)
        new_mapped_items = find_mapped_items_in_representation(new_representation)

        def get_color_maps(items):
            color_maps = []
            for item in items:
                if not item:
                    continue
                if not item.is_a("IfcPolygonalFaceSet") and not item.is_a("IfcTriangulatedFaceSet"):
                    continue
                color_maps.extend(get_colors_from_face_set(item))
            return color_maps

        def get_styled_items(items):
            styled_items = []
            for item in items:
                if not item:
                    continue
                if not item.is_a("IfcFacetedBrep"):
                    continue
                styled_items.extend(get_styled_items_from_faceted_brep(item))
            return styled_items

        changed = self.record_deep_diff(
            global_id, "color_maps_changed", get_color_maps(old_mapped_items), get_color_maps(new_mapped_items)
        )
        changed = (
            self.record_deep_diff(
                global_id,
                "face_styled_items_changed",
                get_styled_items(old_mapped_items),
                get_styled_items(new_mapped_items),
            )
            or changed
        )
        return changed

    def diff_materials(self, old_element, new_element) -> bool:
        old_material_reps = get_material_representations(old_element)
        new_material_reps = get_material_representations(new_element)

        def get_styled_items(representation):
            if not representation:
                return []

            styled_items = []
            for rep in representation.Representations or []:
                if not rep or not rep.is_a("IfcStyledRepresentation"):
                    continue
                for item in rep.Items or []:
                    styled_items.append(item.Styles)
            return styled_items

        old_styled_items = []
        new_styled_items = []
        max_len = max(len(old_material_reps), len(new_material_reps))
        for i in range(max_len):
            old_rep = None if i >= len(old_material_reps) else old_material_reps[i]
            new_rep = None if i >= len(new_material_reps) else new_material_reps[i]
            old_styled_items.extend(get_styled_items(old_rep))
            new_styled_items.extend(get_styled_items(new_rep))

        return self.record_deep_diff(
            new_element.GlobalId, "materials_changed", old_styled_items, new_styled_items
        )

    def diff_shape_aspects(self, global_id, old_representation, new_representation) -> bool:
        def get_shape_aspects(representation, ifc_file):
            return [item for item in ifc_file.get_inverse(representation) if item.is_a("IfcShapeAspect")]

        return self.record_deep_diff(
            global_id,
            "shape_aspects_changed",
            get_shape_aspects(old_representation, self.old),
            get_shape_aspects(new_representation, self.new),
        )

    def diff_styled_items(self, global_id, old_representation, new_representation) -> bool:
        def get_representation_styles(representation, ifc_file):
            styles = []
            for shape_representation in representation.Representations or []:
                for item in shape_representation.Items or []:
                    for styled_item in ifc_file.get_inverse(item):
                        if styled_item.is_a("IfcStyledItem"):
                            styles.append(styled_item.Styles)
            return styles

        return self.record_deep_diff(
            global_id,
            "styled_items_changed",
            get_representation_styles(old_representation, self.old),
            get_representation_styles(new_representation, self.new),
        )

    def diff_presentation_layer_assignment(self, global_id, old_representation, new_representation) -> bool:
        def get_representation_layers(representation, ifc_file):
            return [
                assignment.Name
                for shape_representation in representation.Representations or []
                for assignment in ifc_file.get_inverse(shape_representation)
                if assignment.is_a("IfcPresentationLayerAssignment")
            ]

        return self.record_deep_diff(
            global_id,
            "layer_assignment_changed",
            get_representation_layers(old_representation, self.old),
            get_representation_layers(new_representation, self.new),
        )

    def get_representation_id(self, element: ifcopenshell.entity_instance) -> Union[int, None]:
        if "Representation" not in getattr(element, "__dict__", {}) or not element.Representation:
            return
        for representation in element.Representation.Representations:
            if not representation.is_a("IfcShapeRepresentation"):
                continue
            if representation.RepresentationIdentifier not in ("Body", "Reference"):
                continue
            if representation.RepresentationType == "MappedRepresentation":
                items = representation.Items or []
                if not items:
                    continue
                mapping_source = items[0].MappingSource
                if mapping_source and mapping_source.MappedRepresentation:
                    return mapping_source.MappedRepresentation.id()
            else:
                return representation.id()


class DiffTerminator:
    def match(self, level) -> bool:
        return True

    def give_up_diffing(self, level, diff_instance) -> bool:
        if any(diff_instance.tree.values()):
            raise Exception("Terminated")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show the difference between two IFC files")
    parser.add_argument("old", type=str, help="The old IFC file")
    parser.add_argument("new", type=str, help="The new IFC file")
    parser.add_argument(
        "-o", "--output", type=str, help="The JSON diff file to output. Defaults to no file.", default=None
    )
    parser.add_argument(
        "-r",
        "--relationships",
        type=str,
        help=(
            'A list of space-separated relationships, chosen from "attributes", "geometry", '
            '"type", "property", "container", "aggregate", "classification", "inverse". '
            'Defaults to "attributes geometry inverse" when omitted.'
        ),
        default="",
    )
    args = parser.parse_args()

    print("# IFC Diff")

    start = time.time()
    print("Loading old file ...")
    old = ifcopenshell.open(args.old)
    print("Loading new file ...")
    new = ifcopenshell.open(args.new)

    print("# Loading finished in {:.2f} seconds".format(time.time() - start))
    start = time.time()

    ifc_diff = IfcDiff(
        old,
        new,
        relationships=args.relationships.split() or ["attributes", "geometry", "inverse"],
        is_shallow=False,
        identifiable_classes=["IfcProduct", "IfcProject"],
        comparable_classes=["IfcRoot"],
        inverse_relationship_classes=["IfcRelationship"],
    )
    ifc_diff.diff()

    print("# Diff finished in {:.2f} seconds".format(time.time() - start))
    print("Equal: ", ifc_diff.equal)

    ifc_diff.export(args.output)

    sys.exit(77 if ifc_diff.equal else 0)
