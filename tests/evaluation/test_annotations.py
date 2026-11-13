import pytest

from evaluation.detection.annotations import load_detrac_annotations


def create_xml(tmp_path, content):
    xml_path = tmp_path / "annotations.xml"
    xml_path.write_text(content, encoding="utf-8")
    return xml_path


class TestDetracAnnotationParser:

    def test_loads_frames_and_vehicle_annotations(self, tmp_path):
        xml = """
        <sequence>
            <frame num="1">
                <target_list>
                    <target id="10">
                        <box left="100" top="150"
                             width="80" height="60"/>
                        <attribute vehicle_type="car"
                                   truncation_ratio="0.1"/>
                    </target>
                </target_list>
            </frame>
        </sequence>
        """

        result = load_detrac_annotations(
            create_xml(tmp_path, xml)
        )

        assert list(result.frames) == [1]
        assert result.retained_annotations == 1

        annotation = result.frames[1][0]

        assert annotation.target_id == "10"
        assert annotation.bbox == (100.0, 150.0, 80.0, 60.0)
        assert annotation.vehicle_type == "car"
        assert annotation.occluded is False

    def test_preserves_occlusion_information(self, tmp_path):
        xml = """
        <sequence>
            <frame num="1">
                <target_list>
                    <target id="1">
                        <box left="10" top="20"
                             width="30" height="40"/>
                        <attribute vehicle_type="car"
                                   truncation_ratio="0.1"/>
                        <occlusion/>
                    </target>
                </target_list>
            </frame>
        </sequence>
        """

        result = load_detrac_annotations(
            create_xml(tmp_path, xml)
        )

        assert result.frames[1][0].occluded is True

    def test_excludes_truncation_at_threshold(self, tmp_path):
        xml = """
        <sequence>
            <frame num="1">
                <target_list>
                    <target id="1">
                        <box left="10" top="20"
                             width="30" height="40"/>
                        <attribute vehicle_type="car"
                                   truncation_ratio="0.5"/>
                    </target>
                </target_list>
            </frame>
        </sequence>
        """

        result = load_detrac_annotations(
            create_xml(tmp_path, xml)
        )

        assert result.total_annotations == 1
        assert result.excluded_truncated_annotations == 1
        assert result.retained_annotations == 0

    def test_preserves_original_vehicle_categories(self, tmp_path):
        xml = """
        <sequence>
            <frame num="1">
                <target_list>
                    <target id="1">
                        <box left="10" top="20"
                             width="30" height="40"/>
                        <attribute vehicle_type="others"
                                   truncation_ratio="0.1"/>
                    </target>
                </target_list>
            </frame>
        </sequence>
        """

        result = load_detrac_annotations(
            create_xml(tmp_path, xml)
        )

        assert result.frames[1][0].vehicle_type == "others"

    def test_loads_sequence_level_ignored_regions(self, tmp_path):
        xml = """
        <sequence>
            <ignored_region>
                <box left="10" top="20"
                     width="30" height="40"/>
                <box left="50" top="60"
                     width="70" height="80"/>
            </ignored_region>
        </sequence>
        """

        result = load_detrac_annotations(
            create_xml(tmp_path, xml)
        )

        assert len(result.ignored_regions) == 2
        assert result.ignored_regions[0].bbox == (
            10.0, 20.0, 30.0, 40.0
        )

    def test_preserves_frames_with_no_retained_annotations(self, tmp_path):
        xml = """
        <sequence>
            <frame num="1">
                <target_list>
                    <target id="1">
                        <box left="10" top="20"
                             width="30" height="40"/>
                        <attribute vehicle_type="car"
                                   truncation_ratio="0.8"/>
                    </target>
                </target_list>
            </frame>
        </sequence>
        """

        result = load_detrac_annotations(
            create_xml(tmp_path, xml)
        )

        assert 1 in result.frames
        assert result.frames[1] == ()

    def test_rejects_invalid_truncation_threshold(self, tmp_path):
        with pytest.raises(ValueError):
            load_detrac_annotations(
                tmp_path / "unused.xml",
                max_truncation_ratio=1.5,
            )

    def test_rejects_invalid_bounding_box(self, tmp_path):
        xml = """
        <sequence>
            <frame num="1">
                <target_list>
                    <target id="1">
                        <box left="10" top="20"
                             width="0" height="40"/>
                        <attribute vehicle_type="car"
                                   truncation_ratio="0.1"/>
                    </target>
                </target_list>
            </frame>
        </sequence>
        """

        with pytest.raises(ValueError, match="Non-positive"):
            load_detrac_annotations(
                create_xml(tmp_path, xml)
            )

    def test_missing_annotation_file_raises_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_detrac_annotations(
                tmp_path / "missing.xml"
            )
